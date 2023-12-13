#!/bin/bash
#SBATCH --account def-training-wa
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:3
#SBATCH --tasks-per-node=3
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --output=slurm.%x.%j.out

module load python/3.9
source ../mainenv/bin/activate
wandb offline
export MASTER_ADDR=$(hostname)
echo "$SLURM_NODEID master: $MASTER_ADDR"
echo "ISSLURM_NODEID Launching python script"

srun python ./train.py -init_method tcp://$MASTER_ADDR:3456 -world_size $SLURM_NTASKS
