import argparse
import copy
import math
import os
from pathlib import Path

import torch
from torch import nn

import utils
from multitask_transformer_class import MultitaskTransformerModel


@torch.no_grad()
def prune_two_linear_hidden_mlp(seq: nn.Sequential, new_hidden: int, importance_from: str = "fc3_l1"):
    """
    Prune the shared hidden width of class_net/reg_net.

    Supported layouts:
      Classification head:
        [Linear(emb, h), ReLU, Permute, BN, Permute, Dropout,
         Linear(h, h), ReLU, Permute, BN, Permute, Dropout,
         Linear(h, out)]

      Regression head:
        [Linear(emb, h), ReLU, Permute, BN, Permute,
         Linear(h, h), ReLU, Permute, BN, Permute,
         Linear(h, out)]
    """
    if not isinstance(seq, nn.Sequential):
        raise TypeError("Expected nn.Sequential")

    linear_idxs = [i for i, m in enumerate(seq) if isinstance(m, nn.Linear)]
    if len(linear_idxs) != 3:
        raise ValueError("Expected exactly 3 Linear layers in the task head")

    fc1 = seq[linear_idxs[0]]
    fc2 = seq[linear_idxs[1]]
    fc3 = seq[linear_idxs[2]]

    hidden = fc1.out_features
    if not (0 < new_hidden < hidden):
        raise ValueError(f"new_hidden must be in (0, {hidden}), got {new_hidden}")

    if fc2.in_features != hidden or fc2.out_features != hidden or fc3.in_features != hidden:
        raise ValueError("Task head hidden dimensions do not match the expected shared width")

    if importance_from == "fc3_l1":
        importance = fc3.weight.abs().sum(dim=0)
    elif importance_from == "fc2_out_l1":
        importance = fc2.weight.abs().sum(dim=1)
    elif importance_from == "fc2_in_l1":
        importance = fc2.weight.abs().sum(dim=0)
    else:
        raise ValueError("importance_from must be 'fc3_l1', 'fc2_out_l1', or 'fc2_in_l1'")

    keep = torch.topk(importance, k=new_hidden, largest=True).indices.sort().values

    new_fc1 = nn.Linear(fc1.in_features, new_hidden, bias=fc1.bias is not None)
    new_fc2 = nn.Linear(new_hidden, new_hidden, bias=fc2.bias is not None)
    new_fc3 = nn.Linear(new_hidden, fc3.out_features, bias=fc3.bias is not None)

    new_fc1.weight.copy_(fc1.weight[keep, :])
    if fc1.bias is not None:
        new_fc1.bias.copy_(fc1.bias[keep])

    new_fc2.weight.copy_(fc2.weight[keep][:, keep])
    if fc2.bias is not None:
        new_fc2.bias.copy_(fc2.bias[keep])

    new_fc3.weight.copy_(fc3.weight[:, keep])
    if fc3.bias is not None:
        new_fc3.bias.copy_(fc3.bias)

    seq[linear_idxs[0]] = new_fc1
    seq[linear_idxs[1]] = new_fc2
    seq[linear_idxs[2]] = new_fc3

    return keep


@torch.no_grad()
def prune_tar_net(seq: nn.Sequential, new_hidden: int, importance_from: str = "fc3_l1"):
    """
    Prune the shared hidden width of tar_net:
      [Linear(emb, h), BN(batch), Linear(h, h), BN(batch), Linear(h, input_size)]
    """
    if not isinstance(seq, nn.Sequential):
        raise TypeError("Expected nn.Sequential")

    linear_idxs = [i for i, m in enumerate(seq) if isinstance(m, nn.Linear)]
    if len(linear_idxs) != 3:
        raise ValueError("Expected exactly 3 Linear layers in tar_net")

    fc1 = seq[linear_idxs[0]]
    fc2 = seq[linear_idxs[1]]
    fc3 = seq[linear_idxs[2]]

    hidden = fc1.out_features
    if not (0 < new_hidden < hidden):
        raise ValueError(f"new_hidden must be in (0, {hidden}), got {new_hidden}")

    if fc2.in_features != hidden or fc2.out_features != hidden or fc3.in_features != hidden:
        raise ValueError("tar_net hidden dimensions do not match the expected shared width")

    if importance_from == "fc3_l1":
        importance = fc3.weight.abs().sum(dim=0)
    elif importance_from == "fc2_out_l1":
        importance = fc2.weight.abs().sum(dim=1)
    elif importance_from == "fc2_in_l1":
        importance = fc2.weight.abs().sum(dim=0)
    else:
        raise ValueError("importance_from must be 'fc3_l1', 'fc2_out_l1', or 'fc2_in_l1'")

    keep = torch.topk(importance, k=new_hidden, largest=True).indices.sort().values

    new_fc1 = nn.Linear(fc1.in_features, new_hidden, bias=fc1.bias is not None)
    new_fc2 = nn.Linear(new_hidden, new_hidden, bias=fc2.bias is not None)
    new_fc3 = nn.Linear(new_hidden, fc3.out_features, bias=fc3.bias is not None)

    new_fc1.weight.copy_(fc1.weight[keep, :])
    if fc1.bias is not None:
        new_fc1.bias.copy_(fc1.bias[keep])

    new_fc2.weight.copy_(fc2.weight[keep][:, keep])
    if fc2.bias is not None:
        new_fc2.bias.copy_(fc2.bias[keep])

    new_fc3.weight.copy_(fc3.weight[:, keep])
    if fc3.bias is not None:
        new_fc3.bias.copy_(fc3.bias)

    seq[linear_idxs[0]] = new_fc1
    seq[linear_idxs[1]] = new_fc2
    seq[linear_idxs[2]] = new_fc3

    return keep


def keep_first_n_encoder_layers(transformer_encoder: nn.Module, n: int):
    if not hasattr(transformer_encoder, "layers"):
        raise AttributeError("Expected transformer_encoder.layers")

    layers = transformer_encoder.layers
    total = len(layers)
    if not (1 <= n <= total):
        raise ValueError(f"n must be in [1, {total}], got {n}")

    transformer_encoder.layers = nn.ModuleList(list(layers[:n]))
    transformer_encoder.num_layers = n


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


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


def merge_prop_from_checkpoint(prop, ckpt):
    if not isinstance(ckpt, dict):
        return prop

    saved_prop = ckpt.get("prop")
    if not isinstance(saved_prop, dict):
        return prop

    # Keep current dataset/task choice, but reuse architecture/training-critical values by default.
    for k in [
        "batch", "lr", "nlayers", "emb_size", "nhead", "task_rate", "masking_ratio",
        "lamb", "epochs", "ratio_highest_attention", "avg", "dropout", "nhid",
        "nhid_task", "nhid_tar", "task_type"
    ]:
        if k in saved_prop:
            prop[k] = saved_prop[k]

    return prop


def evaluate_model(model, X_test, y_test, prop, criterion_task):
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


def print_metrics(metrics, task_type):
    if task_type == "classification":
        loss, acc, prec, rec, f1 = metrics
        print(f"Test loss: {loss:.6f}")
        print(f"Test acc:  {acc:.6f}")
        print(f"Test prec: {prec:.6f}")
        print(f"Test rec:  {rec:.6f}")
        print(f"Test f1:   {f1:.6f}")
    else:
        rmse, mae = metrics
        print(f"Test RMSE: {rmse:.6f}")
        print(f"Test MAE:  {mae:.6f}")


def finetune_pruned_model(model, X_train_task, y_train_task, X_test, y_test, prop):
    criterion_tar = torch.nn.MSELoss()
    criterion_task = torch.nn.CrossEntropyLoss() if prop["task_type"] == "classification" else torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=prop["lr"])

    best_state = copy.deepcopy(model.state_dict())
    best_score = -math.inf if prop["task_type"] == "classification" else math.inf

    instance_weights = torch.rand(X_train_task.shape[0], prop["seq_len"], device=prop["device"])

    for epoch in range(1, prop["epochs"] + 1):
        X_train_tar, y_train_tar_masked, y_train_tar_unmasked, boolean_indices_masked, boolean_indices_unmasked = \
            utils.random_instance_masking(
                X_train_task,
                prop["masking_ratio"],
                prop["ratio_highest_attention"],
                instance_weights,
            )

        tar_loss_masked, tar_loss_unmasked, task_loss, instance_weights = utils.multitask_train(
            model,
            criterion_tar,
            criterion_task,
            optimizer,
            X_train_tar,
            X_train_task,
            y_train_tar_masked,
            y_train_tar_unmasked,
            y_train_task,
            boolean_indices_masked,
            boolean_indices_unmasked,
            prop,
        )

        metrics = evaluate_model(model, X_test, y_test, prop, criterion_task)

        if prop["task_type"] == "classification":
            score = metrics[1]
            improved = score > best_score
            extra = f"acc={score:.6f}, f1={metrics[4]:.6f}"
        else:
            score = metrics[0]
            improved = score < best_score
            extra = f"rmse={metrics[0]:.6f}, mae={metrics[1]:.6f}"

        print(
            f"Finetune epoch {epoch}/{prop['epochs']} | "
            f"tar_masked={tar_loss_masked:.6f} | "
            f"tar_unmasked={tar_loss_unmasked:.6f} | "
            f"task_loss={task_loss:.6f} | {extra}"
        )

        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    final_metrics = evaluate_model(model, X_test, y_test, prop, criterion_task)
    return model, final_metrics


def save_pruned_checkpoint(output_path: str, model, prop, source_checkpoint, metrics, args):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "prop": prop,
        "source_checkpoint": args.checkpoint,
        "pruning": {
            "new_nhid_tar": args.new_nhid_tar,
            "new_nhid_task": args.new_nhid_task,
            "new_num_layers": args.new_num_layers,
            "importance_from": args.importance_from,
        },
        "metrics": metrics,
    }

    if isinstance(source_checkpoint, dict):
        payload["source_meta"] = {k: v for k, v in source_checkpoint.items() if k not in ["model_state_dict", "state_dict"]}

    torch.save(payload, output_path)



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
    parser.add_argument("--epochs", type=int, default=20, help="Finetuning epochs after pruning")
    parser.add_argument("--ratio_highest_attention", type=float, default=0.5)
    parser.add_argument("--avg", type=str, default="macro")
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--nhid", type=int, default=128)
    parser.add_argument("--nhid_task", type=int, default=128)
    parser.add_argument("--nhid_tar", type=int, default=128)

    parser.add_argument("--new_nhid_tar", type=int, default=None)
    parser.add_argument("--new_nhid_task", type=int, default=None)
    parser.add_argument("--new_num_layers", type=int, default=None)
    parser.add_argument(
        "--importance_from",
        type=str,
        default="fc3_l1",
        choices=["fc3_l1", "fc2_out_l1", "fc2_in_l1"],
    )
    parser.add_argument("--output", type=str, default="checkpoints/pruned_finetuned_tarnet.pt")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    prop = build_prop_from_args(args)
    source_checkpoint, state_dict = load_checkpoint(args.checkpoint, prop["device"])
    prop = merge_prop_from_checkpoint(prop, source_checkpoint)

    print("Loading and preprocessing data...")
    X_train_task, y_train_task, X_test, y_test, prop = load_and_prepare_data(prop)

    print("Building model...")
    model = build_model(prop)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    params_before = count_params(model)
    print(f"Parameters before pruning: {params_before}")

    if args.new_nhid_tar is not None:
        prune_tar_net(model.tar_net, args.new_nhid_tar, args.importance_from)
        prop["nhid_tar"] = args.new_nhid_tar
        print(f"Pruned tar_net hidden width -> {args.new_nhid_tar}")

    if args.new_nhid_task is not None:
        if prop["task_type"] == "classification":
            prune_two_linear_hidden_mlp(model.class_net, args.new_nhid_task, args.importance_from)
            print(f"Pruned class_net hidden width -> {args.new_nhid_task}")
        else:
            prune_two_linear_hidden_mlp(model.reg_net, args.new_nhid_task, args.importance_from)
            print(f"Pruned reg_net hidden width -> {args.new_nhid_task}")
        prop["nhid_task"] = args.new_nhid_task

    if args.new_num_layers is not None:
        keep_first_n_encoder_layers(model.transformer_encoder, args.new_num_layers)
        prop["nlayers"] = args.new_num_layers
        print(f"Dropped transformer_encoder layers -> {args.new_num_layers}")

    model = model.to(prop["device"])

    params_after = count_params(model)
    print(f"Parameters after pruning:  {params_after}")
    print(f"Reduction: {100.0 * (1.0 - params_after / params_before):.2f}%")

    print("Finetuning pruned model...")
    model, final_metrics = finetune_pruned_model(model, X_train_task, y_train_task, X_test, y_test, prop)

    print("Final test metrics:")
    print_metrics(final_metrics, prop["task_type"])

    save_pruned_checkpoint(args.output, model, prop, source_checkpoint, final_metrics, args)
    print(f"Saved pruned + finetuned checkpoint to: {args.output}")


if __name__ == "__main__":
    main()
