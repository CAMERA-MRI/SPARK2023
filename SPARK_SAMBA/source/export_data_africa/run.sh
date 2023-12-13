#!/bin/bash
#SBATCH --account def-training-wa
#SBATCH --gpus-per-node=t4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00

module load python/3.9
source ../mainenv/bin/activate
python flair_y.py