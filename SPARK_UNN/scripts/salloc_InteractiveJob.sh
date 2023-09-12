#!/bin/bash


python mlcube.py infer --data_path=/scratch/guest187/Data/data_cube --output_path=/scratch/guest187/Results/results_mlcube_ftSSA --ckpts_path=/scratch/guest187/MLCubes/working/sparkUNN_mlcube/mlcube/workspace/additional_files/checkpoints --parameters_file=/scratch/guest187/MLCubes/working/sparkUNN_mlcube/mlcube/workspace/parameters.yaml

########## USE THE FOLLOWING COMMANDS TO CREATE AN INTERACTIVE SESSSION TO TEST CODE ######### 
###### salloc --account=def-training-wa --time=hh:mm:ss --cpus-per-task=[3-22] --mem-per-cpu=[8-12G] --gres=gpu:t4:[1-3] ###########
######          other options include: --nodes[1-3], --gpus-per-task=t4:[1-3] --ntasks=[2?]                              ###########
###################################################################################################
###### EXAMPLE USES:
        # salloc --time=3:0:0 --cpus-per-task=12 --mem-per-cpu=8G --ntasks=2 --account=def-training-wa --nodes 2
        # salloc --time=0:10:00 --gpus-per-node=t4:2 --cpus-per-task=8 --mem=64G --account=def-training-wa --nodes 2
        # salloc --time=01:00:00 --gpus-per-node=t4:2 --cpus-per-task=6 --mem-per-cpu=8G --account=def-training-wa
        # salloc --time=1:00:00 --gpus-per-node=v100:1 --cpus-per-task=3 --mem=64G --ntasks=3 --account=def-training-wa
        # salloc --time=03:00:00 --cpus-per-task=12 --mem-per-cpu=8G --account=def-training-wa --gpus-per-node=t4:1

###############################################################
################### DIRECTORY PATHS ###########################
# data_dir=/scratch/guest187/Data/train_all
# results_dir=/scratch/guest187/Data/train_all/results
# chkpt_store=/scratch/guest187/Data/train_all/checkpoints
# git=/home/guest187/GitRepo_Brats23/UNN_BraTS23/scripts

# FOR MONAI trainer testing use
salloc --time=02:00:00 --gpus-per-node=t4:1 --cpus-per-task=6 --mem-per-cpu=8G --account=def-training-wa

module load python/3.9
module load cuda/11.0
source /home/guest187/hackathon/bin/activate

data_dir=/scratch/guest187/Data/train_all/train_data
results_dir=/scratch/guest187/Data/train_all/results/test_run
git=/home/guest187/GitRepo_Brats23/UNN_BraTS23/scripts


python $git/monai_trainer.py --seed 42 --data $data_dir --results $results_dir --epochs 2 --gpus 1 --run_name "tester" --data_used "SSA" --criterion "dice" --batch_size=2

#------------------ USING JUPYTER ----------------
salloc --time=03:00:00 --cpus-per-task=12 --mem-per-cpu=8G --account=def-training-wa --gpus-per-node=t4:1

salloc --time=03:00:00 --cpus-per-task=6 --mem-per-cpu=8G --account=def-training-wa --gpus-per-node=v100:1 --constraint=cascade,v100
module load python/3.9
source /home/guest187/hackathon/bin/activate

# I suggest using jupyter lab
srun $VIRTUAL_ENV/bin/jupyterlab.sh

## but you can also:
srun $VIRTUAL_ENV/bin/notebook.sh

## the following is the same regardless:
# once the above runs, it will give you a bunch of urls
# open local terminal and follow instructions from https://docs.alliancecan.ca/wiki/Advanced_Jupyter_configuration

# e.g. http://gra1156.graham.sharcnet:8888/lab?token=ef6a0f72cabe151aa7dbc808f0b52a2e13e7b35ca7c36a95
# In your local terminal type:
        # For mac/linux: sshuttle --dns -Nr <username>@<cluster>.computecanada.ca
        sshuttle --dns -Nr guest187@graham.computecanada.ca
                ## paste url that looks like this in your browser:
                http://gra1154.graham.sharcnet:8888/lab?token=e717e3ccab3c0664a46be3bd29fdfb047e9a6e9417bfac96

        # For windows  ssh -L 8888:<hostname:port> <username>@<cluster>.computecanada.ca
        ssh -L 8888:gra1162.graham.sharcnet:8888 guest187@graham.computecanada.ca
                # on chrome/firefox type: http://localhost:8888/?token=<token>
                http://localhost:8888/?token=b60f351f238d9abd066e5877b7fdb84096a45a50f22b69ea

                # OR this one which works better with jupyter lab
                http://127.0.0.1:8888/lab?token=1b3f59a629180f81f366eb98d0f0c3659f12a9b02d4b2b1

http://127.0.0.1:8888/lab?token=e717e3ccab3c0664a46be3bd29fdfb047e9a6e9417bfac96





import nibabel as nib
import numpy as np
gli_pred188 = nib.load("/scratch/guest187/Results/results_mlcube_gli/run1/predictions/BraTS-SSA-00188-000.nii.gz")
gli_pred169 = nib.load("/scratch/guest187/Results/results_mlcube_gli/run1/predictions/BraTS-SSA-00169-000.nii.gz")
ft_pred188 = nib.load("/scratch/guest187/Results/results_mlcube_ftSSA/run1/predictions/BraTS-SSA-00188-000.nii.gz")
ft_pred169 = nib.load("/scratch/guest187/Results/results_mlcube_ftSSA/run1/predictions/BraTS-SSA-00169-000.nii.gz")
ft_val169 = nib.load("/scratch/guest187/Results/train_gli/preds/train_gli_ftSSA_valSSA/BraTS-SSA-00169-000.nii.gz")
ft_val188 = nib.load("/scratch/guest187/Results/train_gli/preds/train_gli_ftSSA_valSSA/BraTS-SSA-00188-000.nii.gz")
gli_val188 = nib.load("/scratch/guest187/Results/train_gli/preds/train_gli_valSSA/BraTS-SSA-00188-000.nii.gz")
gli_val169 = nib.load("/scratch/guest187/Results/train_gli/preds/train_gli_valSSA/BraTS-SSA-00169-000.nii.gz")
gli_pred188rep = nib.load("/scratch/guest187/Results/results_mlcube_gli/run2/predictions/BraTS-SSA-00188-000.nii.gz")
glipred169rep = nib.load("/scratch/guest187/Results/results_mlcube_gli/run2/predictions/BraTS-SSA-00169-000.nii.gz")
ft_pred188rep = nib.load("/scratch/guest187/Results/results_mlcube_gli/run2/predictions/BraTS-SSA-00188-000.nii.gz")
ft_pred169rep = nib.load("/scratch/guest187/Results/results_mlcube_gli/run2/predictions/BraTS-SSA-00169-000.nii.gz")


pred_list = [gli_pred188,gli_pred188rep,gli_val188,gli_pred169,gli_val169,glipred169rep]
pred_list = [ft_pred188,ft_val188,ft_pred169,ft_val169,ft_pred188rep,ft_pred169rep]

gli_pred188_np = np.array(gli_pred188.dataobj)
gli_pred188rep_np = np.array(gli_pred188rep.dataobj)
gli_val188_np = np.array(gli_val188.dataobj)
gli_pred169_np = np.array(gli_pred169.dataobj)
gli_val169_np = np.array(gli_val169.dataobj)
gli_pred169rep_np = np.array(glipred169rep.dataobj)

ft_pred188_np = np.array(ft_pred188.dataobj)
ft_val188_np = np.array(ft_val188.dataobj)
ft_pred169_np = np.array(ft_pred169.dataobj)
ft_val169_np = np.array(ft_val169.dataobj)
ft_pred169rep_np = np.array(ft_pred169rep.dataobj)
ft_pred188rep_np = np.array(ft_pred188rep.dataobj)

predXval_ft188 = np.sum(ft_pred188_np != ft_val188_np)
pred2Xval_ft188 =np.sum(ft_pred188rep_np != ft_val188_np)
predXpred2_ft188 = np.sum(ft_pred188_np != ft_pred188rep_np)

predXval_ft169 = np.sum(ft_pred169_np != ft_val169_np)
pred2Xval_ft169 = np.sum(ft_pred169rep_np != ft_val169_np)
predXpred2_ft169 = np.sum(ft_pred169_np != ft_pred169rep_np)

predXval_gli188 = np.sum(gli_pred188_np != gli_val188_np)
pred2Xval_gli188 = np.sum(gli_pred188rep_np != gli_val188_np)
predXpred2_gli188 = np.sum(gli_pred188_np != gli_pred188rep_np)

predXval_gli169 = np.sum(gli_pred169_np != gli_val169_np)
pred2Xval_gli169 = np.sum(gli_pred169rep_np != gli_val169_np)
predXpred2_gli169 = np.sum(gli_pred169_np != gli_pred169rep_np)

pred_ftXgli188 = np.sum(ft_pred188_np != gli_pred188_np)
val_ftXgli188 = np.sum(ft_val188_np != gli_val188_np)
pred2_ftXgli188 = np.sum(ft_pred169rep_np != gli_pred188rep_np)

pred_ftXgli188 = np.sum(ft_pred169_np != gli_pred169_np)
val_ftXgli188 = np.sum(ft_val169_np != gli_val169_np)
pred2_ftXgli188 = np.sum(ft_pred169rep_np != gli_pred169rep_np)



print("predXval_ft188 = ", predXval_ft188," pred2Xval_ft188 = ", pred2Xval_ft188,"predXpred2_ft188 = ",predXpred2_ft188)
print("predXval_ft169 = ",predXval_ft169 ," pred2Xval_ft169 = ", pred2Xval_ft169,"predXpred2_ft169 = ", predXpred2_ft169)
print("predXval_gli188 = ",predXval_gli188,"pred2Xval_gli188 = ", pred2Xval_gli188, "predXpred2_gli188 = ", predXpred2_gli188)
print("predXval_gli169 = ",predXval_gli169,"pred2Xval_gli169 = ", pred2Xval_gli169, "predXpred2_gli169 = ", predXpred2_gli169)
print("pred_ftXgli188 = ",pred_ftXgli188,"val_ftXgli188 = ", val_ftXgli188, "pred2_ftXgli188 = ", pred2_ftXgli188)
print("pred_ftXgli188 = ",pred_ftXgli188,"val_ftXgli188 = ", val_ftXgli188, "pred2_ftXgli188 = ", pred2_ftXgli188)



predXval_ft188 =  5802819  pred2Xval_ft188 =  5807258 predXpred2_ft188 =  6106

predXval_ft169 =  5880537  pred2Xval_ft169 =  5888878 predXpred2_ft169 =  15279

predXval_gli188 =  5802041 pred2Xval_gli188 =  5802041 predXpred2_gli188 =  0

predXval_gli169 =  5874636 pred2Xval_gli169 =  5874636 predXpred2_gli169 =  1

pred_ftXgli188 =  15280 val_ftXgli188 =  15280 pred2_ftXgli188 =  0
