#!/bin/bash
#SBATCH --account def-training-wa
#SBATCH --gpus-per-node=t4:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load gcc/9.3.0 opencv python/3.9 scipy-stack
source ../mainenv/bin/activate
wandb offline
python yolo_v8_z_no_black.py