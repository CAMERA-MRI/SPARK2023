#!/bin/bash
#SBATCH --account def-uanazodo
#SBATCH --gpus-per-node=t4:1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=model1_bts.out

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

module load python/3.9
source /home/guest189/hackathon/bin/activate

# python "/home/guest189/Optimized U-Net/Optimized U-Net/preprocess.py" --task 11 --ohe --exec_mode training
# python '/home/guest189/Optimized U-Net/Optimized U-Net/preprocess.py' --task 12 --ohe --exec_mode test

# python '/home/guest189/Optimized U-Net/Optimized U-Net/main.py' --data /scratch/guest189/BraTS_2021_data/BraTS2021_train/11_3d --results /scratch/guest189/BraTS_2021_data/BraTS2021_train/results_modelbts --ckpt_store_dir /scratch/guest189/BraTS_2021_data/BraTS2021_train/results_modelbts --brats --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 30 --fold 0 --amp --gpus 1 --task 11 --save_ckpt
# python '/home/guest189/Optimized U-Net/Optimized U-Net/main.py' --data /scratch/guest189/BraTS_2021_data/BraTS2021_train/output_folder --results /scratch/guest189/BraTS_2021_data/BraTS2021_train/results_model --ckpt_store_dir /scratch/guest189/BraTS_2021_data/BraTS2021_train/results_model --brats --deep_supervision --depth 3 --filters 16 32 64 128 --min_fmap 2 --scheduler --learning_rate 0.0005 --epochs 100 --fold 0 --amp --gpus 1 --task 11 --save_ckpt

python '/home/guest189/Optimized U-Net/Optimized U-Net/main.py' --data /scratch/guest189/BraTS_2021_data/BraTS2021_train/output_folder --brats --brats22_model --scheduler --learning_rate 0.0003 --epochs 10 --fold 0 --amp --gpus 1 --task 11 --nfolds 10 --save_ckpt
# python '/home/guest189/Optimized U-Net/Optimized U-Net/main.py' --gpus 1 --amp --save_preds --exec_mode predict --brats --brats22_model --data /data/12_3d/test --ckpt_path /results/checkpoints/epoch=8-dice=89.94.ckpt --tta
# python '/home/guest189/Optimized U-Net/Optimized U-Net/main.py' --brats --deep_supervision --depth 6 --filters 64 96 128 192 256 384 512 --min_fmap 2 --scheduler --learning_rate 0.0003 --epochs 30 --fold 0 --amp --gpus 1 --task 11 --save_ckpt    --constraint=cascade,v100