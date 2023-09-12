#!/bin/bash
#SBATCH --account=def-training-wa
# SBATCH --reservation hackathon-wr-gpu
#SBATCH --nodes=1
# SBATCH --gpus-per-node=t4:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:15:00

source ./SWIN_ENV/bin/activate

# # remote cluster 

# # local computer
path_swin='/home/odcus/Software/Kilimanjaro_swinUNETR/'
path_data='/home/odcus/Data/BraTS_Africa_data/'
export PYTHONPATH=$path_swin


# # generate model predictions 
python $path_swin'test.py'  --infer_overlap=0.7\
 --data_dir=$path_data --exp_name='epoch100_baseModel_GLI_test'\
 --json_list=$path_swin'jsons/brats23_gli_test.json'\
 --pretrained_dir=$path_swin'pretrained_models/'\
 --pretrained_model_name='model-epoch100-baseModel-2023.pt'

# run quality assurance on model predictions
# python $path_swin'kilimajaro_scripts/quality_assurance.py'