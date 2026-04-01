#!/bin/bash
#SBATCH --job-name=tarnet
#SBATCH --output=tarnet_train.slurmlog
#SBATCH --error=tarnet_train.slurmlog
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p gpu

srun python script.py --dataset BasicMotions --task_type classification --batch 32 --epochs 100
srun python prune_finetune_test.py --checkpoint checkpoints/BasicMotions_best.pt --dataset BasicMotions --batch 32 --new_nhid_tar 64 --new_nhid_task 64 --epochs 10 --output checkpoints/BasicMotions_pruned_finetuned.pt
