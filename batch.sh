#!/bin/bash
#SBATCH --job-name=tarnet
#SBATCH --output=tarnet_train.slurmlog
#SBATCH --error=tarnet_train.slurmlog
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p gpu

srun python script.py --dataset AF --task_type classification --batch 32
srun python prune_finetune_test.py --checkpoint checkpoints/AF_best.pt --dataset AF --batch 32 --new_nhid_tar 64 --new_nhid_task 64 --new_num_layers 1 --epochs 20 --output checkpoints/AF_pruned_finetuned.pt
