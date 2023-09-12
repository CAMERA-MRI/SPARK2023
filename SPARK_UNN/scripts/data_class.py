
import torch
import os
import numpy as np
from utils.utils import get_main_args
# from torch.utils.data import Dataset
from monai.data import Dataset
import torch.utils.data as data_utils
import nibabel as nib
import torchio as tio
import logging

class MRIDataset(Dataset):
    """
    Given a set of images and corresponding labels (i.e, will give it all training images + labels, and same for val and test)
    folder structure: subjectID/subjectID-stk.npy, -lbl.npy (i.e. contains 2 files)
    """

    def __init__(self, data_dir, data_folders, transform=None, SSAtransform=None):
            self.data_folders = data_folders                            # path for each data folder in the set
            self.transform = transform
            self.SSAtransform = SSAtransform
            self.imgs = []                                              # store images to load (paths)
            self.lbls = []   
            self.mode = None                                            # store corresponding labels (paths)
            for img_folder in self.data_folders:                        # run through each subjectID folder
                folder_path = os.path.join(data_dir, img_folder)                                                            
                self.SSA = True if 'SSA' in img_folder else False       # check if current file is from SSA dataset       
                for file in os.listdir(folder_path):                    # check folder contents
                    if os.path.isfile(os.path.join(folder_path, file)):
                        if file.endswith("-lbl.npy"):
                            self.lbls.append(os.path.join(folder_path, file))   # Save segmentation mask (file path)
                            self.mode = "labels"
                        elif file.endswith("-stk.npy"):
                            self.imgs.append(os.path.join(folder_path, file))   # Save image (file path)

    def __len__(self):
        # Return the amount of images in this set
        return len(self.imgs)
    
    def __getitem__(self, idx):
        logger = logging.getLogger(__name__)

        name = os.path.dirname(self.imgs[idx])
        # Load files
        image = np.load(self.imgs[idx])
        image = torch.from_numpy(image) # 4, 240, 240, 155
        # logger.info(f"image shape is {image.shape}")
        if self.mode is not None:
            mask = np.load(self.lbls[idx])
            mask = torch.from_numpy(mask) # 240, 240, 155
            # logger.info(f"mask file exists; mask shape is {mask.shape}")

        # logger.info(self.imgs[idx] )
        # logger.info("========================")
        # logger.info(self.lbls[idx] )
        # logger.info("========================")           

        if self.transform is not None: # Apply general transformations
        # transforms such as crop, flip, rotate etc will be applied to both the image and the mask
            if self.mode is not None:
                subject = tio.Subject(
                    image=tio.ScalarImage(tensor=image),
                    label=tio.LabelMap(tensor=mask),
                    name=name
                    )
                # Apply transformation to GLI data to reduce quality (creating fake SSA data)                    
                tranformed_subject = self.transform(subject)
                if self.SSA == False and self.SSAtransform is not None:
                    tranformed_subject = self.SSAtransform(tranformed_subject)
                logger.info(f"Tranformed_subject: {tranformed_subject['name']}")

                image = tranformed_subject["image"].data
                mask = tranformed_subject["label"].data
                # logger.info(f"Image shape is {image.shape}")
                # logger.info(f"Mask shape is {mask.shape}")

                return image, mask, self.imgs[idx]
            
            else:
                subject = tio.Subject(
                    image=tio.ScalarImage(tensor=image),
                    name=name
                    )
                tranformed_subject = self.transform(subject)           
                logger.info(f"Tranformed_subject: {tranformed_subject['name']}")
                image = tranformed_subject["image"].data
                return image, self.imgs[idx]
    
    def get_paths(self):
        return self.img_pth, self.seg_pth
    
    def get_subj_info(self):
        return self.subj_dir_pths, self.subj_dirs
        #, self.SSA
    
    def get_transforms(self):
        return self.transform

'''
--------------- CHECK WITH ALEX -----------------------
        if self.mode == "labels":
            mask = np.load(self.lbls[idx])
            mask = torch.from_numpy(mask) # 240, 240, 155

        # print(self.imgs[idx] )
        # print("========================")
        # print(self.lbls[idx] )
        # print("========================")           

        if self.transform is not None: # Apply general transformations
        # transforms such as crop, flip, rotate etc will be applied to both the image and the mask
            if self.mode == "labels":
                subject = tio.Subject(
                    image=tio.ScalarImage(tensor=image),
                    mask=tio.LabelMap(tensor=mask)
                    )
                tranformed_subject = self.transform(subject)
                # Apply transformation to GLI data to reduce quality (creating fake SSA data)
                if self.SSA == False and self.SSAtransform is not None:
                    tranformed_subject = self.SSAtransform(tranformed_subject)
            
                print("Tranformed_subject: ", tranformed_subject)
                image = tranformed_subject["image"].data
                mask = tranformed_subject["mask"].data
                return image, mask, self.imgs[idx]
            else:
                subject = tio.Subject(
                    image=tio.ScalarImage(tensor=image),
                    )
                tranformed_subject = self.transform(subject)           
                print("Tranformed_subject: ", tranformed_subject)
                image = tranformed_subject["image"].data
                return image, self.imgs[idx]  
'''
############ THIS SECTION WILL BE REMOVED ##########
"""
AA __init__ : delete below once copied to local system

    def __init__(self, subj_dirL, task=args.task, modalities=args.modal, transform=None, SSAtransform=None):
        self.data_dir = subj_dirL # path for each subject folder in the set
        self.modalities = modalities
        self.task = task

        self.transform = transform
        self.SSAtransform = SSAtransform
        
        self.subj_dirs, self.subjIDls, self.subj_dir_pths = [],[],[]
        # store images to load (paths)
        self.img_pth, self.proc_imgs, self.imgs_npy = [],[],[]
        # store corresponding labels (paths)
        self.seg_pth, self.proc_lbls, self.lbls_npy = [],[],[]

        file_ext_dict_prep = {
            "-seg.nii.gz": self.seg_pth,
            "-lbl.nii.gz": self.proc_lbls,
            "-stk.nii.gz": self.proc_imgs,
            **{f"-{m}.nii.gz": self.img_pth for m in modalities},
        }

        file_ext_dict_aug = {
            "-lbl.npy": self.imgs_npy,
            "-stk.npy": self.lbls_npy
        }

        for root, dirs, files in os.walk(self.data_dir):
            for directory in sorted(dirs, key=lambda x: x.lower(), reverse=True):
                if not "BraTS-" in directory:
                    break
                else:
                    self.subj_dirs.append(str(directory))
                    self.subj_dir_pths.append(os.path.join(root,directory))
                    #self.subjIDls.append(self.subj_dirs)
                    self.SSA = 'SSA' in self.subj_dirs
            for file in files:
                file_pth = os.path.join(root, file)
                if os.path.isfile(file_pth) and task=='data_prep':
                    for ext, list_to_append in file_ext_dict_prep.items():
                        if file.endswith(ext):
                            #print(file_pth)
                            list_to_append.append(file_pth)
                else:
                    for ext, list_to_append in file_ext_dict_aug.items():
                        if file.endswith(ext):
                            #print(file_pth)
                            list_to_append.append(file_pth)
"""