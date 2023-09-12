A modified brain tumor segmentation model based on 'Extending nn-UNet for Brain Tumor Segmentation' for Brats-Africa challenge
* The major modification is on Data augmentation, In addition to the previous augmentation techniques we have added a custom image sharpenning step, unsharp masking. (training.data_augmentation.custom_transform.unsharp_masking_lucy)
* Trainer class: trainer.competition_with_custome_Trainers.Brats2023Lucy.nnUNetTrainerV2BraTSRegions_moreDA.nnUNetTrainerV2BraTSRegions_DA4_BN_BD_UNSM_LUCY
As stated on the paper "The Brain Tumor Segmentation (BraTS) Challenge 2023: Glioma Segmentation in Sub-Saharan Africa Patient Population (BraTS-Africa)", one of the challenges of this dataset is the quality due to the instruments used to capture the images.


This model is under-developement.

* Installation
- Get inside the folder "Brats-Africa_Eth_Lucy_team_spark"
    $ pip install e .
* Training

    $ nnUNet_train 3d_fullres nnUNetTrainerV2BraTSRegions_DA4_BN_BD_UNSM_LUCY 500 <Folds> --npz



