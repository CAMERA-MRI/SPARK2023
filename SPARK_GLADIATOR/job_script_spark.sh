#!/bin/bash
#SBATCH --account def-training-wa
#SBATCH --nodes=2
#SBATCH --gpus-per-node=t4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00

module load python/3.9
source /home/guest188/hackathon/bin/activate
# srun python prepare_dataset.py
# srun python preprocess.py --task 11 --ohe --exec_mode training
# srun python preprocess.py --task 13 --ohe --exec_mode training
# srun python preprocess.py --task 12 --ohe --exec_mode test
# srun python main.py --brats --deep_supervision --depth 7 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 10 --fold 0 --amp --gpus 1 --task 11  --save_ckpt
srun python main.py --brats --deep_supervision --optimizer adam --depth 7 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 112 --data /scratch/guest188/BraTS_Africa_data/results/13_3d --fold 0 --exec_mode train --amp --gpus 1 --task 13 --save_ckpt 
# srun python main.py --brats --deep_supervision --optimizer adam --depth 7 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 140 --data /scratch/guest188/BraTS_Africa_data/results/11_3d --fold 0 --exec_mode train --amp --gpus 2 --task 11 --save_ckpt 
# srun python main.py --gpus 1 --amp --save_preds --depth 7 --filters 64 96 128 192 256 384 512 --min_fmap 2 --exec_mode predict --brats --data /scratch/guest188/BraTS_Africa_data/results/12_3d/test --task 12 --ckpt_path /scratch/guest188/BraTS_Africa_data/checkpoints/epoch=111-dice=89.23.ckpt --results /scratch/guest188/BraTS_Africa_data/val_results --tta 
# srun python main.py --gpus 1 --amp --save_preds --exec_mode predict --brats --data /scratch/guest182/BraTS_Challenge_Africa_Data/12_3d/test --task 12 --ckpt_path /scratch/guest182/BraTS_Challenge_Africa_Data/results/checkpoints/epoch=9-dice=86.34.ckpt --tta
# srun python postprocess.py 
