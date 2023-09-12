# MICCAI 2023 Brain Tumour Segmentation Challenge: Team UNN *(Umuntu Ngumuntu Ngabantu)*
## Using this repository
This repository contains newly developed code as well as slightly modified version of the Optimised U-Net model as found in the Nvidia Deep Learning Examples repository for [nnUnet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet)

BraTS2023 final submission only uses the Optimised Unet framework. 4 main models were trained and evaluated:
    1. training with only SSA data
    2. training with only BraTS-GLIOMA data
    3. training with SSA and BraTS-GLIOMA data combined
    3. training with only BraTS-GLIOMA data and then fine-tuning the model with SSA data

The newly developed code was created in order to further explore the effect of additional data augmentations on the generalisability of this framework. The aim was also to simplify the process and reduce computational capacity such that the model can be run from low resource settings possibly using open source computing infrastructure. These are found in the folder scripts, as well as within several notebooks. Due to time, resource and capacity contraints, this was not completed in the 2 month period of the challenge. 
## Scripts in this repository
Team UNN code: All folders except for OptiNet contain original code authored by SPARK team UNN. Below are some of the main scripts.
### Notebooks
Below outlines key notebooks created for use during EDA, script development and evaluations. 
```
├──notebooks
  Explorations and code used for challenge submission
    └──data-exploration.ipynb
        Key for initial EDA of the multimodal data provided for the BraTS2023 challenge. Provides visuals of all subjects across all planes (when N is 60)
    └──inference_opti.ipynb
        Final steps of inference for validation submissions. Contains postprocessing steps taken and visualisations of predicted segments
    └──prepdata-exploration.ipynb
        Used to double check data prep script created
  Explorations for modifying OptiNet to allow for more generalised training to SSA data:
    └──transforms-exploration.ipynb
        Initial exploration of potential transforms functions, and working with torchio subjects
        Requires local copy of data
        └──transforms-exploration_SynapseData.ipynb
        Visually explore the effects of trasformations on the original data. To assist with determining which additional augmentations are likely to improve model.
        Requires synapse login.
    └──transforms-exploration_MRS+SNP.ipynb
        Exploring augmentations from literature that directly apply formula to volume intensities
        Requires synapse login
    └──dataloader_exploration.ipynb
        Used for checking data loader function
    └──usefulFx.ipynb
  └─playground
        └──Augmentations_Efficient.ipynb
        └──test_scripts.ipynb
  └─results
    These files are used for comparing validation submissions to determine differences in model performance.
        └──Box_Whisker.ipynb
        └──inference_monai.ipynb
            This code is to be used in conjunction with the team UNN created training scripts
        └──inference_opti.ipynb
            This is the code used to generate final nifti files for submission during validation phase
        └──predictions_SynapseData.ipynb
        └──Statistics.ipynb
        └──Statistics_F_statistic.ipynb
```
### Modified Model Training
1. data_prep : is an independent script, meant to only be run once to preprocess (and save) all raw data provided from the challenge, so that the data is ready to be put into dataloaders

2. data_loader : reads from data_class and data_transforms, gets the data ready for model (will be read in from training script)

3. training script: The training and validation data from the data_loader are passed to the trainer, respectively.

4. Inference: the trained model is used to make predictions (i.e segmentating the brain tumor). 
```
|--scripts
    |--This folder contains all scripts created for modified training procedures. Attempts to successfully deploy OptiNet 2022 model with the code is currently unsuccessful
    |--data_preparation_OptiNet.py
        |--duplication of Futrega et al. data preparation code as found in their notebook "Brats22.ipynb"
    |--trainer.py
        |--main call to run all trainig procedures.
        |--calls on other scripts within folder
        |--monai_functions.py defines all functions used for training and validation. This script calls helper functions from the utils folder.
    |-- inference.py
        |--incomplete, refer to notebook playground
```
### Compute Canada
**Compute Canada ~bashrc aliases for quick access:**
`
scr="cd /scratch/guest187"
data="cd /scratch/guest187/Data"
results="cd /scratch/guest187/Results"
home="cd /home/guest187"
oN="cd /home/guest187/BraTS23_SSA/train/OptiUnet_run"
mon="cd /home/guest187/BraTS23_SSA/train/MON_UNet"
gh="cd /home/guest187/GitRepo_Brats23/UNN_BraTS23"
ghON="cd /home/guest187/GitRepo_Brats23/nnUnet"
`
## Folder & File Structure Requirements
### Training & Validation Data
Total file: Training data
- BraTS glioma dataset = 1251
- BraTS SSA dataset = 60

All data files are labelled as follows in the new data release:
- BraTS-GLI-#####-000-#.nii.gz
- BraTS-SSA-#####-000-#.nii.gz

data preparation outputs: 
    - data/train/subj/
        - subjxxx-stk.nii : the stacked volumes from T1n, T1c, T2w, T2f (in that order)
        - subjxxx-stk.npy : initial pre-processed stacked files == check RAS and normalise; main pre-process must include croporpad
        - subjxxx-lbl.nii : extracted seg file img data
        - subjxxx-lbl.npy : initial pre-processed seg file == MUST MATCH stk.npy transformations AT ALL TIMES

data augmentation outputs: currently nothing is saved
    - BRaTS23 GLI TRAIN DATA - some augmented to mimic poor quality data from SSA (RandomFlip, RandomAffine)
    - BraTS23 GLI (fake SSA) - to make fake SSA data, RandomAnisotropy, Randomblur, RandomNoise, RandomMotion, RandomBiasField, and 
      RandomGhosting was applied to GLI data. 
    - BraTS23 SSA TRAIN DATA - Currently no augmentation done
    - BraTS23 GLI VALIDATION DATA - should be changed to BraTS fake SSA data to be used as a validation set
    - 

### Submission: Validation
Refer to Challenge page on Synapse for submission requirements
The segmentation files need to adhere to the following rules:
- Be NIfTI and use .nii.gz extension
- Dimensions should be 240 X 240 X 155 and origin at [0, -239, 0]
- Use CaPTk to verify
- Filenames should end with 5 digits case ID, followed by a dash, and 3-digit timepoint. (eg. *{ID}-{timepoint}.nii.gz

Segmentations should be contained in a zip or tarball archive and upload this to Synapse
To submit click: File Tools > Submit File to Challenge
There are 5 queues and as a team we are limited to 2 submissions per day


