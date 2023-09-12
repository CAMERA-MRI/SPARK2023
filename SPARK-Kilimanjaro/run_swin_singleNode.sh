#!/bin/bash
#SBATCH --account=def-training-wa
#SBATCH --nodes=1
#SBATCH --gpus-per-node=t4:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:00:00
module load python/3.9
source /home/guest183/run_swinUNETR_kilimanjaro/SWIN_ENV/bin/activate

# # Code to generate a joson file for a given training or testing data folder
python kfold_json_generator.py 


# Brats 2021
# path_swin='/home/guest183/research-contributions/SwinUNETR/BRATS21/'
# path_data='/scratch/guest183/BraTS_2021_data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/'
# # code to run the main method of the SWIN UNER network
# # echo $path_swin'main.py'
# python $path_swin'main.py' --distributed --lrschedule='cosine_anneal' --json_list=$path_swin'jsons/brats21_folds.json' --sw_batch_size=8 --batch_size=2 --data_dir=$path_data --val_every=20 --infer_overlap=0.7 --out_channels=3 --in_channels=4 --spatial_dims=3 --save_checkpoint --use_checkpoint --feature_size=48 --max_epochs=60 --logdir='epoch_60_gpu_4t4_2021Data'

# Brats 2023 Africa
path_swin='/home/guest183/research-contributions/SwinUNETR/BRATS21/'
path_data='/scratch/guest183/BraTS_Africa_data/'
# code to run the main method of the SWIN UNER network
# echo $path_swin'main.py'
# # to train on multiple GPUs
python $path_swin'main.py'\
 --pretrained_dir=$path_swin'pretrained_models/'\
 --pretrained_model_name='model-epoch100-baseModel-2023.pt'\
 --resume_ckpt --save_checkpoint --use_checkpoint\
 --json_list=$path_swin'jsons/brats23_africa_folds.json'\
 --distributed --data_dir=$path_data\
 --sw_batch_size=8 --batch_size=2 --infer_overlap=0.7\
 --max_epochs=100 --val_every=25 --lrschedule='warmup_cosine'\
 --out_channels=3 --in_channels=4 --spatial_dims=3\
 --feature_size=48 --logdir='epoch200_baseModel_resumeCheckpointAfter100Epochs'

# --resume_ckpt --pretrained_dir=$path_swin'runs/4_gpu_60_epochs/' --pretrained_model_name='model_final.pt'
# 

# --roi_x=128 --roi_y=128 --roi_z=128

# --cache_dataset --save_checkpoint

