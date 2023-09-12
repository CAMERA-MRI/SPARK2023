
import os
import numpy as np
import nibabel as nib
def compare (path1, path2):
    ar1 = nib.load(path1)
    ar1_fdata = ar1.get_fdata()
    ar1_np = np.array(ar1_fdata)
    print("Min fdata: ", np.min(ar1_fdata), "Min np:, ", np.min(ar1_np))
    print("Max fdata: ", np.max(ar1_fdata), "Max np:, ", np.max(ar1_np))
    
    ar2 = nib.load(path2)
    ar2_fdata = ar2.get_fdata()
    ar2_np = np.array(ar2_fdata)
    print("Min fdata: ", np.min(ar2_fdata), "Min np:, ", np.min(ar2_np))
    print("Max fdata: ", np.max(ar2_fdata), "Max np:, ", np.max(ar2_np))

    print("{} voxels are the same".format(np.sum(ar1_fdata == ar2_fdata)))
    print("{} voxels are different".format(np.sum(ar1_fdata != ar2_fdata)))
    print("=================== {} ===================".format(os.path.basename(path1)))



# #------------------------LOAD NIFTI------------------
# #-------predictions run 1
# gli_pred188 = nib.load("/scratch/guest187/Results/results_mlcube_gli/run1/predictions/BraTS-SSA-00188-000.nii.gz")
# gli_pred169 = nib.load("/scratch/guest187/Results/results_mlcube_gli/run1/predictions/BraTS-SSA-00169-000.nii.gz")

# ft_pred188 = nib.load("/scratch/guest187/Results/results_mlcube_ftSSA/run1/predictions/BraTS-SSA-00188-000.nii.gz")
# ft_pred169 = nib.load("/scratch/guest187/Results/results_mlcube_ftSSA/run1/predictions/BraTS-SSA-00169-000.nii.gz")

# # --------validation submissions
# gli_val188 = nib.load("/scratch/guest187/Results/train_gli/preds/train_gli_valSSA/BraTS-SSA-00188-000.nii.gz")
# gli_val169 = nib.load("/scratch/guest187/Results/train_gli/preds/train_gli_valSSA/BraTS-SSA-00169-000.nii.gz")

# ft_val188 = nib.load("/scratch/guest187/Results/train_gli/preds/train_gli_ftSSA_valSSA/BraTS-SSA-00188-000.nii.gz")
# ft_val169 = nib.load("/scratch/guest187/Results/train_gli/preds/train_gli_ftSSA_valSSA/BraTS-SSA-00169-000.nii.gz")

# # ------------ predictions run 2
# gli_pred188rep = nib.load("/scratch/guest187/Results/results_mlcube_gli/run2/predictions/BraTS-SSA-00188-000.nii.gz")
# glipred169rep = nib.load("/scratch/guest187/Results/results_mlcube_gli/run2/predictions/BraTS-SSA-00169-000.nii.gz")

# ft_pred188rep = nib.load("/scratch/guest187/Results/results_mlcube_gli/run2/predictions/BraTS-SSA-00188-000.nii.gz")
# ft_pred169rep = nib.load("/scratch/guest187/Results/results_mlcube_gli/run2/predictions/BraTS-SSA-00169-000.nii.gz")

# #------------------------CONVERT NPY------------------
# gli_pred188_np = np.array(gli_pred188.dataobj)
# gli_pred188rep_np = np.array(gli_pred188rep.dataobj)
# gli_val188_np = np.array(gli_val188.dataobj)
# gli_pred169_np = np.array(gli_pred169.dataobj)
# gli_val169_np = np.array(gli_val169.dataobj)
# gli_pred169rep_np = np.array(glipred169rep.dataobj)

# ft_pred188_np = np.array(ft_pred188.dataobj)
# ft_val188_np = np.array(ft_val188.dataobj)
# ft_pred169_np = np.array(ft_pred169.dataobj)
# ft_val169_np = np.array(ft_val169.dataobj)
# ft_pred169rep_np = np.array(ft_pred169rep.dataobj)
# ft_pred188rep_np = np.array(ft_pred188rep.dataobj)

# predXval_ft188 = np.sum(ft_pred188_np != ft_val188_np)
# pred2Xval_ft188 = np.sum(ft_pred188rep_np != ft_val188_np)
# predXpred2_ft188 = np.sum(ft_pred188_np != ft_pred188rep_np)

# predXval_ft169 = np.sum(ft_pred169_np != ft_val169_np)
# pred2Xval_ft169 = np.sum(ft_pred169rep_np != ft_val169_np)
# predXpred2_ft169 = np.sum(ft_pred169_np != ft_pred169rep_np)

# predXval_gli188 = np.sum(gli_pred188_np != gli_val188_np)
# pred2Xval_gli188 = np.sum(gli_pred188rep_np != gli_val188_np)
# predXpred2_gli188 = np.sum(gli_pred188_np != gli_pred188rep_np)

# predXval_gli169 = np.sum(gli_pred169_np != gli_val169_np)
# pred2Xval_gli169 = np.sum(gli_pred169rep_np != gli_val169_np)
# predXpred2_gli169 = np.sum(gli_pred169_np != gli_pred169rep_np)

# pred_ftXgli188 = np.sum(ft_pred188_np != gli_pred188_np)
# val_ftXgli188 = np.sum(ft_val188_np != gli_val188_np)
# pred2_ftXgli188 = np.sum(ft_pred169rep_np != gli_pred188rep_np)

# pred_ftXgli169 = np.sum(ft_pred169_np != gli_pred169_np)
# val_ftXgli169 = np.sum(ft_val169_np != gli_val169_np)
# pred2_ftXgli169 = np.sum(ft_pred169rep_np != gli_pred169rep_np)



# print("predXval_ft188 = ", predXval_ft188," pred2Xval_ft188 = ", pred2Xval_ft188,"predXpred2_ft188 = ",predXpred2_ft188)
# print("predXval_ft169 = ",predXval_ft169 ," pred2Xval_ft169 = ", pred2Xval_ft169,"predXpred2_ft169 = ", predXpred2_ft169)

# print("predXval_gli188 = ",predXval_gli188,"pred2Xval_gli188 = ", pred2Xval_gli188, "predXpred2_gli188 = ", predXpred2_gli188)
# print("predXval_gli169 = ",predXval_gli169,"pred2Xval_gli169 = ", pred2Xval_gli169, "predXpred2_gli169 = ", predXpred2_gli169)

# print("pred_ftXgli188 = ",pred_ftXgli188,"val_ftXgli188 = ", val_ftXgli188, "pred2_ftXgli188 = ", pred2_ftXgli188)
# print("pred_ftXgli169 = ",pred_ftXgli169,"val_ftXgli169 = ", val_ftXgli169, "pred2_ftXgli169 = ", pred2_ftXgli169)



# predXval_ft188 =  5802819  pred2Xval_ft188 =  5807258 predXpred2_ft188 =  6106
# predXval_ft169 =  5880537  pred2Xval_ft169 =  5888878 predXpred2_ft169 =  15279

# predXval_gli188 =  5802041 pred2Xval_gli188 =  5802041 predXpred2_gli188 =  0
# predXval_gli169 =  5874636 pred2Xval_gli169 =  5874636 predXpred2_gli169 =  1

# pred_ftXgli188 =  6106 val_ftXgli188 =  6106 pred2_ftXgli188 =  1108201
# pred_ftXgli169 =  15280 val_ftXgli169 =  15280 pred2_ftXgli169 =  0

