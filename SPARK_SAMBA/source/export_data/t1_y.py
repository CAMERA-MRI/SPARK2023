# Saving t1_z  #approved
print("t1_y no_image")

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

def squarify(M,val = 0):
    (a,b)=M.shape
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    return np.pad(M,padding,mode='constant',constant_values=val)

# create_folders
base = "/scratch/guest190/"
os.makedirs(base + "BraTS_data", exist_ok=True)


# prepare SAM model
model_type = 'vit_h'
checkpoint = 'sam_vit_h_4b8939.pth'
sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to('cuda')


os.makedirs(base + "BraTS_data/t1/y", exist_ok=True)
os.makedirs(base + "BraTS_data/t1/y/images", exist_ok=True)
os.makedirs(base + "BraTS_data/t1/y/embeddings", exist_ok=True)

src_folder = "/scratch/guest190/kaggle/"
image_size = 256


files = os.listdir(src_folder)
for i in tqdm(files):
  img =  nib.load(src_folder + i + "/"+ i +"_t1.nii.gz")
  img = img.get_fdata()
  num_slices = img.shape[1]

  three_d_embedding = []
  for s in range(num_slices):
    image = img[:,s,:]
    # np.save(base + "BraTS_data/t1/y/images/"+i+"_"+ str(s), image)
    image = squarify(image)
    # plt.imshow(image, cmap ="gray")
    # plt.show()

    lower_bound, upper_bound = np.percentile(image, 0.5), np.percentile(image, 99.5)
    image_data_pre = np.clip(image, lower_bound, upper_bound)
    d = (np.max(image_data_pre)-np.min(image_data_pre))
    d = 1 if d==0 else d
    image_data_pre = ((image_data_pre - np.min(image_data_pre))/d) *255.0
    image_data_pre[image==0] = 0
    image = transform.resize(image_data_pre[:,:], (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
    image = np.stack((image, image, image), axis = -1 )
    image = np.uint8(image)

    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    image = sam_transform.apply_image(image)
    # resized_shapes.append(resize_img.shape[:2])
    image = torch.as_tensor(image.transpose(2, 0, 1)).to('cuda')
    # model input: (1, 3, 1024, 1024)
    image = sam_model.preprocess(image[None,:,:,:]) # (1, 3, 1024, 1024)
    image = image[0,...] # (3, 1024, 1024)
    # plt.imshow(image[1,:,:].cpu(), cmap ="gray")
    # plt.show()
    with torch.no_grad():
      embedding = sam_model.image_encoder(image[None,...])[0].cpu().numpy()
    three_d_embedding.append(embedding)
  np.save(base + "BraTS_data/t1/y/embeddings/" +i, np.array(three_d_embedding))
  # print( np.array(three_d_embedding).shape)
