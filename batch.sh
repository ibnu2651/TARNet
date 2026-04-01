#!/bin/bash
#SBATCH --job-name=tarnet
#SBATCH --output=tarnet_train.slurmlog
#SBATCH --error=tarnet_train.slurmlog
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p gpu

srun python script.py --dataset AF --task_type classification --batch 8