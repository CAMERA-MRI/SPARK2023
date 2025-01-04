import os
from torch.utils.data import Dataset as dataset_torch
import numpy as np
import random
import torch
import SimpleITK as sitk


mean = np.array([0.5,])
std = np.array([0.5,])

def _make_dataset(dir):
   images = []
   for root, _, fnames in sorted(os.walk(dir)):
       for fname in fnames:
           # if is_image_file(fname):
           path = os.path.join(root, fname)
           item = path
           images.append(item)
   return images

def _make_image_namelist(dir):
   images = []
   namelist = []
   for root, _, fnames in sorted(os.walk(dir)):
       for fname in fnames:
           if fname.endswith('_t1.nii.gz'):
               item_name = fname
               namelist.append(item_name)
               item_path = os.path.join(root, fname)
               images.append(item_path)
   return images, namelist


class data_set(dataset_torch):
   def __init__(self, root, split='train'):
       self.root = root
       assert split in ('train', 'val', 'test')
       self.split = split
       self.imgs, self.nlist = _make_image_namelist(self.root)
       self.epi = 0
       self.img_num = len(self.imgs)

   def __len__(self):
       return len(self.imgs)

   def __getitem__(self, index):
       path = self.imgs[index]
       case_name = self.nlist[index]

       path_t1 = self.imgs[index]
       path_t2 = path_t1.replace('_t1.nii.gz', '_t2.nii.gz')
       path_t1ce = path_t1.replace('_t1.nii.gz', '_t1ce.nii.gz')
       path_flair = path_t1.replace('_t1.nii.gz', '_flair.nii.gz')


       path_label = path_t1.replace('_t1.nii.gz', '_seg.nii.gz')

       t1 = sitk.ReadImage(path_t1)
       t2 = sitk.ReadImage(path_t2)
       t1ce = sitk.ReadImage(path_t1ce)
       flair = sitk.ReadImage(path_flair)

       label = sitk.ReadImage(path_label)

       t1, t2, t1ce, flair = sitk.GetArrayFromImage(t1), sitk.GetArrayFromImage(t2), sitk.GetArrayFromImage(t1ce), sitk.GetArrayFromImage(flair)
       label = sitk.GetArrayFromImage(label)


       ##### here to place preprocessing steps ####

       t1, t2, t1ce, flair = np.expand_dims(t1, 0), np.expand_dims(t2, 0), np.expand_dims(t1ce, 0), np.expand_dims(flair, 0)
       t1, t2, t1ce, fliar = np.array(t1, float), np.array(t2, float), np.array(t1ce, float), np.array(flair, float)
       label = np.expand_dims(label, 0)
       label = np.array(label, float)
       print("t1 shape", t1.shape)

       imgs = np.concatenate((t1, t2, t1ce, flair), 0)

       return imgs, label