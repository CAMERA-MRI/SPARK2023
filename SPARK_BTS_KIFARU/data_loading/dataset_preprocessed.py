import os
from torch.utils.data import Dataset as dataset_torch
import numpy as np
import random
import torch


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
           if fname.endswith('_x.npy'):
               current_path = os.path.join(root, fname)
               namelist.append(fname)
               item_path = os.path.join(root, fname)
               images.append(item_path)
   return images, namelist


class data_set(dataset_torch):
   def __init__(self, root, split='train'):
       self.root = root
       assert split in ('train', 'val', 'test')
       self.split = split
       self.imgs, self.nlist = _make_image_namelist(self.root)


       self.img_num = len(self.imgs)


   def __len__(self):
       return len(self.imgs)


   def __getitem__(self, index):
       path_img = self.imgs[index]


       path_label = path_img.replace('_x.npy', '_y.npy')
       img = np.load(path_img)
       label = np.load(path_label)
       return img, label