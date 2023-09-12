""" This script is to prepare the provided data set for pre-processing and then run all pre-processing.
It will take as input a directory path to the training data and apply the following:
    1. Read in the dataset folder structure
    2. Store variables
        subjID = subject IDs
        img_dir = path to each imaging modality
        lbl dir = path to corresponding segmentation mask
    3. Load nifty file for each modality
    4. Extract voxel intensity values, header information and affine matrix
        Stack voxel data from each modality into 1 tensor
        cropping out background
        Add an extra channel for one hot encoding??
    5. Save the following files:
        images/subjIDxxx-stk.nii.gz = stacked modalities output into a nifti file in an images folder
        labels/subjIDxxx-lbl.nii.gz = segmentation mask
    6. Create and save json file that contains a dictionary of dictionaries and lists:
        A dictionary of dummy coding for seg labels as provided by BraTS
            "labels" : {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"}
        A dictionary of dummy coding for each modality
            "modality": {"0": "FLAIR", "1": "T1", "2": "T1CE", "3": "T2"}
        A dictionary of dictionaries containing the image-label path pairs
            "training": [{"image": "images/subjIDxxx.nii.gz", "label": "labels/subjIDxxx_seg.nii.gz"}

Add noise defs for fake SSA data in an if 
"""

## Import key libraries
import os
from glob import glob
import json
import time
from subprocess import call
import logging
from joblib import Parallel, delayed

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import nibabel as nib
import numpy as np
import torch

import torchio as tio

import utils.utils
from utils.utils import get_main_args
from data_transforms import transforms_preproc
from data_transforms import define_transforms

def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)

def run_parallel(func, args):
    return Parallel(n_jobs=os.cpu_count())(delayed(func)(arg) for arg in args)

import itertools
import json
import math
import os
import pickle

import monai.transforms as transforms
import nibabel
import numpy as np
from joblib import Parallel, delayed
from skimage.transform import resize
import utils.utils


class Preprocessor:
    def __init__(self, args):
        self.args = args
        self.target_spacing = None
        self.task = args.task
        self.task_code = args.data_grp
        self.training = args.preproc_set == "training"
        self.data_path = args.data
        metadata_path = os.path.join(self.data_path, "dataset.json")
        self.patch_size = [192, 224, 160]
        self.metadata = json.load(open(metadata_path, "r"))
        self.crop_foreg = transforms.CropForegroundd(keys=["image", "label"], source_key="image")
        nonzero = True  # normalize only non-zero region for MRI
        self.normalize_intensity = transforms.NormalizeIntensity(nonzero=nonzero, channel_wise=True)
        if self.args.preproc_set == "val":
            dataset_json = json.load(open(metadata_path, "r"))
            dataset_json["val"] = dataset_json["training"]
            with open(metadata_path, "w") as outfile:
                json.dump(dataset_json, outfile)
        self.data_grp = args.data_grp

    def run(self):
        print(f"Preprocessing {self.data_path}")
        self.collect_spacings()
        # if self.verbose:
        #     print(f"Target spacing {self.target_spacing}")
        self.run_parallel(self.preprocess_pair, self.args.preproc_set)
        pickle.dump(
            {
                "spacings": self.target_spacing,
                "n_class": len(self.metadata["labels"]),
                "in_channels": len(self.metadata["modality"]) + int(self.args.ohe),
            },
            open(os.path.join(self.results, "config.pkl"), "wb"),
        )

    def preprocess_pair(self, pair):
        fname = os.path.basename(pair["image"] if isinstance(pair, dict) else pair)
        image, label, image_spacings = self.load_pair(pair)

        # Crop foreground and store original shapes.
        orig_shape = image.shape[1:]
        transStd, transCrp, oheZN = transforms_preproc(target_shape=self.target_shape)
        fakeSSA = tio.Compose([
            transforms.RandomRotation((0, 180)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            tio.OneOf([
                tio.transforms.RandomBlur(std=(0.5, 1.5)),
                tio.transforms.RandomNoise(mean=0, std=(0, 0.33)),
                tio.transforms.RandomMotion(num_transforms=3, image_interpolation='nearest'),
                tio.transforms.RandomBiasField(coefficients=1),
                tio.transforms.RandomGhosting(intensity=1.5)], p=0.8)])
        if self.trans == 'imageCP':
            image = transCrp(image)
            print('after spatial cropping image :', image.shape)
        elif self.trans == 'imageOHE':
            image = oheZN(oheZN)
        else:
            image = transStd(image)        
        # bbox = transforms.utils.generate_spatial_bounding_box(image)
        # print('bbox :', bbox)
        # print('base image :', image.shape)
        # image = transforms.SpatialCrop(roi_start=bbox[0], roi_end=bbox[1])(image)
        image_metadata = np.vstack([self.target_shape, orig_shape, image.shape[1:]])
        if label is not None:
            if self.trans == 'imageCP':
                label = transCrp(label)
            elif self.trans == 'imageOHE':
                label = oheZN(label)
            else:
                label = transStd(image)
        if self.data_grp == "fSSATr":
            image = fakeSSA(image)
            label = fakeSSA(label)

        self.save_npy(label, fname, "Orig-lbl.npy")
        print("Saved original label")

        if self.training:
            image, label = self.standardize(image, label)

        self.save(image, label, fname, image_metadata)
        print("Image, label saved")

    def standardize(self, image, label):
        pad_shape = self.calculate_pad_shape(image)
        image_shape = image.shape[1:]
        if pad_shape != image_shape:
            paddings = [(pad_sh - image_sh) / 2 for (pad_sh, image_sh) in zip(pad_shape, image_shape)]
            image = self.pad(image, paddings)
            label = self.pad(label, paddings)


    def save(self, image, label, fname, image_metadata):
        self.save_npy(image, fname, f"{self.data_grp}-stk.npy")
        if label is not None:
            self.save_npy(label, fname, f"{self.data_grp}-lbl.npy")
        if image_metadata is not None:
            self.save_npy(image_metadata, fname, f"{self.data_grp}_meta.npy")

    def load_pair(self, pair):
        image = self.load_nifty(pair["image"] if isinstance(pair, dict) else pair)
        image_spacing = self.load_spacing(image)
        image = image.get_fdata().astype(np.float32)
        image = self.standardize_layout(image)
        if self.training:
            label = self.load_nifty(pair["label"]).get_fdata().astype(np.uint8)
            label = self.standardize_layout(label)
        else:
            label = None
        return image, label, image_spacing

    def calculate_pad_shape(self, image):
        min_shape = self.patch_size[:]
        image_shape = image.shape[1:]
        if len(min_shape) == 2:  # In 2D case we don't want to pad depth axis.
            min_shape.insert(0, image_shape[0])
        pad_shape = [max(mshape, ishape) for mshape, ishape in zip(min_shape, image_shape)]
        return pad_shape

    def get_spacing(self, pair):
        image = nibabel.load(os.path.join(self.data_path, pair["image"]))
        spacing = self.load_spacing(image)
        return spacing

    def collect_spacings(self):
        spacing = self.run_parallel(self.get_spacing, "training")
        spacing = np.array(spacing)
        target_spacing = np.median(spacing, axis=0)
        if max(target_spacing) / min(target_spacing) >= 3:
            lowres_axis = np.argmin(target_spacing)
            target_spacing[lowres_axis] = np.percentile(spacing[:, lowres_axis], 10)
        self.target_spacing = list(target_spacing)


    def save_npy(self, image, fname, suffix):
        print(os.path.join(self.results, fname.replace(".nii.gz", suffix)))
        # np.save(os.path.join(self.results, fname.replace(".nii.gz", suffix)), image, allow_pickle=False)

    def run_parallel(self, func, exec_mode):
        return Parallel(n_jobs=self.args.n_jobs)(delayed(func)(pair) for pair in self.metadata[exec_mode])

    def load_nifty(self, fname):
        return nibabel.load(os.path.join(self.data_path, fname))

    @staticmethod
    def load_spacing(image):
        return image.header["pixdim"][1:4].tolist()[::-1]

    @staticmethod
    def pad(image, padding):
        pad_d, pad_w, pad_h = padding
        return np.pad(
            image,
            (
                (0, 0),
                (math.floor(pad_d), math.ceil(pad_d)),
                (math.floor(pad_w), math.ceil(pad_w)),
                (math.floor(pad_h), math.ceil(pad_h)),
            ),
        )

    @staticmethod
    def standardize_layout(data):
        if len(data.shape) == 3:
            data = np.expand_dims(data, 3)
        return np.transpose(data, (3, 2, 1, 0))
    
    def transforms_preproc(target_shape=False):
    
        to_ras = tio.ToCanonical() # reorient to RAS+
        # resample_t1space = tio.Resample(image_interpolation='nearest') # target output space (ie. match T2w to the T1w space) 
        if target_shape != False:
            target_shape=(192, 224, 160)
            crop_pad = tio.CropOrPad(target_shape)
        else:
            crop_pad = None
        one_hot_enc = tio.OneHot(num_classes=4)
        normalise_foreground = tio.ZNormalization(masking_method=lambda x: x > x.float().mean()) # threshold values above mean only, for binary mask
        # masked = tio.Mask(masking_method=tio.LabelMap(label))
        normalise = tio.ZNormalization()
        oheZN = tio.Compose([one_hot_enc, normalise_foreground])

        transStd = tio.Compose(to_ras, normalise)
        transCrp = tio.Compose(to_ras, crop_pad, normalise)

        return transStd, transCrp, oheZN


def main():
    logging.basicConfig(filename='04-07_data_transforms.log', filemode='w', level=logging.DEBUG)

    args = get_main_args()
    startT = time.time()
    logging.info("Beginning Preprocessing.")

    Preprocessor(args).run()
   
    endT= time.time()
    logging.info(f"Data Processing complete. Total time taken: {(endT - startT):.2f}")
        
if __name__=='__main__':
    main()