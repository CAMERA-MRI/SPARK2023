# Saving flair_z  #approved
print("masks no_image")

import os
import torch
import numpy as np

from tqdm import tqdm

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import nibabel as nib
from skimage import transform



# from SurfaceDice import compute_dice_coefficient

# set seeds
torch.manual_seed(2023)
np.random.seed(2023)


# create_folders
base = "/scratch/guest190/"
os.makedirs(base + "BraTS_data_africa", exist_ok=True)


os.makedirs(base + "BraTS_data_africa/mask/z", exist_ok=True)
os.makedirs(base + "BraTS_data_africa/mask/x", exist_ok=True)
os.makedirs(base + "BraTS_data_africa/mask/y", exist_ok=True)

src_folder = "/scratch/guest190/africa/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData/"
image_size = 256

files = os.listdir(src_folder)
for i in tqdm(files):
  img =  nib.load(src_folder + i + "/"+ i +"-seg.nii.gz")
  img = img.get_fdata()
  num_slices_z = img.shape[2]
  num_slices_x = img.shape[0]
  num_slices_y = img.shape[1]



  for s in range(num_slices_z):
    image = img[:,:,s]
    np.save(base + "BraTS_data_africa/mask/z/"+i+"_"+ str(s), image)
  
  for s in range(num_slices_x):
    image = img[s,:,:]
    np.save(base + "BraTS_data_africa/mask/x/"+i+"_"+ str(s), image)

  for s in range(num_slices_y):
    image = img[:,s,:]
    np.save(base + "BraTS_data_africa/mask/y/"+i+"_"+ str(s), image)
