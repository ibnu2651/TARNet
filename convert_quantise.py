import argparse
import copy
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader

from onnxruntime.quantization import quantize_dynamic, QuantType

import utils
from multitask_transformer_class import MultitaskTransformerModel


class ClassificationWrapper(nn.Module):
    """
    Wraps MultitaskTransformerModel so ONNX sees a single-tensor forward.
    Your original model is called as: model(x, "classification")
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, "classification")


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("Unsupported checkpoint format")

    return ckpt, state_dict


def maybe_convert_class_targets(y_train_task, y_test, task_type):
    if task_type != "classification":
        return y_train_task, y_test, None

    if y_train_task.ndim > 1:
        y_train_task = torch.argmax(y_train_task, dim=1)
    if y_test.ndim > 1:
        y_test = torch.argmax(y_test, dim=1)

    nclasses = int(torch.max(y_train_task).item() + 1)
    return y_train_task.long(), y_test.long(), nclasses


def build_prop_from_args(args):
    prop = utils.get_prop(args)
    prop["device"] = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    return prop


def merge_prop_from_checkpoint(prop, ckpt):
    if not isinstance(ckpt, dict):
        return prop

    saved_prop = ckpt.get("prop")
    if not isinstance(saved_prop, dict):
        return prop

    for k in [
        "masking_ratio", "task_rate",
        "lamb", "ratio_highest_attention", "avg", "dropout",
        "task_type",
        "emb_size", "nhead", "nhid",
        "nhid_tar", "nhid_task", "nlayers",
    ]:
        if k in saved_prop:
            prop[k] = saved_prop[k]

    return prop


def load_and_prepare_data(prop):
    data_path = f"./data/{prop['dataset']}/"
    X_train, y_train, X_test, y_test = utils.data_loader(prop["dataset"], data_path, prop["task_type"])
    X_train_task, y_train_task, X_test, y_test = utils.preprocess(prop, X_train, y_train, X_test, y_test)

    y_train_task, y_test, nclasses = maybe_convert_class_targets(y_train_task, y_test, prop["task_type"])

    prop["seq_len"] = X_train_task.shape[1]
    prop["input_size"] = X_train_task.shape[2]
    prop["nclasses"] = nclasses if prop["task_type"] == "classification" else None

    return X_train_task, y_train_task, X_test, y_test, prop


def build_model(prop):
    return MultitaskTransformerModel(
        prop["task_type"],
        prop["device"],
        prop["nclasses"],
        prop["seq_len"],
        prop["batch"],
        prop["input_size"],
        prop["emb_size"],
        prop["nhead"],
        prop["nhid"],
        prop["nhid_tar"],
        prop["nhid_task"],
        prop["nlayers"],
        prop["dropout"],
    ).to(prop["device"])


def evaluate_pt_model(model, X_test, y_test, prop):
    criterion_task = torch.nn.CrossEntropyLoss() if prop["task_type"] == "classification" else torch.nn.MSELoss()
    metrics = utils.test(
        model,
        X_test,
        y_test,
        prop["batch"],
        prop["nclasses"],
        criterion_task,
        prop["task_type"],
        prop["device"],
        prop["avg"],
    )
    return metrics


def export_to_onnx(model, sample_input, fp32_path, opset=17):
    model.eval()

    wrapper = ClassificationWrapper(model).eval()
    wrapper.cpu()

    sample_input = sample_input.cpu().float()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            sample_input,
            fp32_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {
                    1: "seq_len",
                },
                "output": {
                    0: "batch_size",
                },
            },
        )

    onnx_model = onnx.load(fp32_path)
    onnx.checker.check_model(onnx_model)
    print(f"Exported FP32 ONNX model to: {fp32_path}")


def quantize_onnx(fp32_path, int8_path):
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )
    print(f"Exported INT8 ONNX model to: {int8_path}")


def run_onnx_inference(ort_session, batch_tensor):
    batch_np = batch_tensor.cpu().numpy().astype(np.float32)
    ort_inputs = {
        ort_session.get_inputs()[0].name: batch_np
    }
    out = ort_session.run(None, ort_inputs)[0]
    return torch.from_numpy(out)


def compare_pytorch_vs_onnx(model, fp32_path, int8_path, X_test, y_test, prop):
    model.eval()

    n = min(X_test.shape[0], y_test.shape[0])
    X_test = X_test[:n]
    y_test = y_test[:n]

    ort_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    ort_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    pt_outputs = []
    onnx_fp32_outputs = []
    onnx_int8_outputs = []

    fixed_batch = prop["batch"]

    for start in range(0, X_test.shape[0], fixed_batch):
        batch = X_test[start:start + fixed_batch]

        pt_out = run_pytorch_batch(model, batch, prop["device"], fixed_batch)
        onnx_fp32_out = run_onnx_batch(ort_fp32, batch, fixed_batch)
        onnx_int8_out = run_onnx_batch(ort_int8, batch, fixed_batch)

        pt_outputs.append(pt_out)
        onnx_fp32_outputs.append(onnx_fp32_out)
        onnx_int8_outputs.append(onnx_int8_out)

    pt_outputs = torch.cat(pt_outputs, dim=0)
    onnx_fp32_outputs = torch.cat(onnx_fp32_outputs, dim=0)
    onnx_int8_outputs = torch.cat(onnx_int8_outputs, dim=0)
    y_test = y_test.cpu()

    print("pt_outputs shape:", pt_outputs.shape)
    print("onnx_fp32_outputs shape:", onnx_fp32_outputs.shape)
    print("onnx_int8_outputs shape:", onnx_int8_outputs.shape)
    print("y_test shape:", y_test.shape)

    fp32_max_abs_diff = torch.max(torch.abs(pt_outputs - onnx_fp32_outputs)).item()
    int8_max_abs_diff = torch.max(torch.abs(pt_outputs - onnx_int8_outputs)).item()

    print(f"Max abs diff (PyTorch vs ONNX FP32): {fp32_max_abs_diff:.8f}")
    print(f"Max abs diff (PyTorch vs ONNX INT8): {int8_max_abs_diff:.8f}")

    pt_preds = pt_outputs.argmax(dim=1)
    onnx_fp32_preds = onnx_fp32_outputs.argmax(dim=1)
    onnx_int8_preds = onnx_int8_outputs.argmax(dim=1)

    pt_acc = (pt_preds == y_test).float().mean().item()
    onnx_fp32_acc = (onnx_fp32_preds == y_test).float().mean().item()
    onnx_int8_acc = (onnx_int8_preds == y_test).float().mean().item()

    print(f"PyTorch accuracy:   {pt_acc * 100:.4f}%")
    print(f"ONNX FP32 accuracy: {onnx_fp32_acc * 100:.4f}%")
    print(f"ONNX INT8 accuracy: {onnx_int8_acc * 100:.4f}%")


def benchmark_pytorch(model, X_test, device, runs=100):
    import time

    model.eval()
    batch = X_test[:model.batch] if hasattr(model, "batch") else X_test[:32]
    batch = batch.to(device).float()

    times = []

    with torch.no_grad():
        for _ in range(10):
            _ = model(batch, "classification")

        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            _ = model(batch, "classification")

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)

    times = torch.tensor(times)
    return times.mean().item(), times.median().item()


def benchmark_onnx(onnx_path, X_test, batch_size=32, runs=100):
    import time

    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    batch = X_test[:batch_size].cpu().float().numpy().astype(np.float32)
    input_name = ort_session.get_inputs()[0].name

    times = []

    for _ in range(10):
        _ = ort_session.run(None, {input_name: batch})

    for _ in range(runs):
        start = time.time()
        _ = ort_session.run(None, {input_name: batch})
        end = time.time()
        times.append(end - start)

    times = torch.tensor(times)
    return times.mean().item(), times.median().item()


def run_pytorch_batch(model, batch, device, fixed_batch):
    actual_bs = batch.shape[0]
    batch = batch.to(device).float()

    if actual_bs < fixed_batch:
        pad = batch[:1].repeat(fixed_batch - actual_bs, 1, 1)
        batch = torch.cat([batch, pad], dim=0)

    with torch.no_grad():
        out, _ = model(batch, "classification")
        out = out[:actual_bs].cpu()

    return out


def run_onnx_batch(ort_session, batch, fixed_batch):
    actual_bs = batch.shape[0]
    batch = batch.cpu().float()

    if actual_bs < fixed_batch:
        pad = batch[:1].repeat(fixed_batch - actual_bs, 1, 1)
        batch = torch.cat([batch, pad], dim=0)

    batch_np = batch.numpy().astype(np.float32)
    input_name = ort_session.get_inputs()[0].name
    out = ort_session.run(None, {input_name: batch_np})[0]
    out = torch.from_numpy(out[:actual_bs])

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="AF")
    parser.add_argument("--task_type", type=str, default="classification", choices=["classification", "regression"])

    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--task_rate", type=float, default=0.5)
    parser.add_argument("--masking_ratio", type=float, default=0.15)
    parser.add_argument("--lamb", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ratio_highest_attention", type=float, default=0.5)
    parser.add_argument("--avg", type=str, default="macro")
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--nhid", type=int, default=128)
    parser.add_argument("--nhid_task", type=int, default=128)
    parser.add_argument("--nhid_tar", type=int, default=128)

    parser.add_argument("--onnx_fp32", type=str, default="checkpoints/tarnet_fp32.onnx")
    parser.add_argument("--onnx_int8", type=str, default="checkpoints/tarnet_int8.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    prop = build_prop_from_args(args)
    source_checkpoint, state_dict = load_checkpoint(args.checkpoint, prop["device"])
    print(source_checkpoint.get("prop", {}))
    prop = merge_prop_from_checkpoint(prop, source_checkpoint)

    print("Loading and preprocessing data...")
    X_train_task, y_train_task, X_test, y_test, prop = load_and_prepare_data(prop)

    print("Building model...")
    model = build_model(prop)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Evaluating PyTorch model...")
    pt_metrics = evaluate_pt_model(model, X_test, y_test, prop)
    print("PyTorch test metrics:", pt_metrics)

    model = model.to("cpu").eval()
    wrapper = ClassificationWrapper(model).to("cpu").eval()
    sample_input = X_test[:prop["batch"]].to("cpu").float()
    export_to_onnx(model, sample_input, args.onnx_fp32, opset=args.opset)
    quantize_onnx(args.onnx_fp32, args.onnx_int8)

    model = model.to(prop["device"]).eval()

    print("Comparing PyTorch vs ONNX...")
    compare_pytorch_vs_onnx(model, args.onnx_fp32, args.onnx_int8, X_test, y_test, prop)

    print("=== Inference Benchmark ===")
    pt_mean, pt_median = benchmark_pytorch(model, X_test, prop["device"])
    print(f"PyTorch -> mean: {pt_mean:.6f}s, median: {pt_median:.6f}s")

    onnx_fp32_mean, onnx_fp32_median = benchmark_onnx(args.onnx_fp32, X_test, batch_size=prop["batch"])
    print(f"ONNX FP32 -> mean: {onnx_fp32_mean:.6f}s, median: {onnx_fp32_median:.6f}s")

    onnx_int8_mean, onnx_int8_median = benchmark_onnx(args.onnx_int8, X_test, batch_size=prop["batch"])
    print(f"ONNX INT8 -> mean: {onnx_int8_mean:.6f}s, median: {onnx_int8_median:.6f}s")


if __name__ == "__main__":
    main()