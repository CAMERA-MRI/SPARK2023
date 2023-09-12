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
import itertools

import nibabel as nib
import numpy as np
import torch

import torchio as tio

from utils.utils import get_main_args
from data_transforms import transforms_preproc
from data_transforms import define_transforms


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)
    
# def run_parallel(func, *args):
#     return Parallel(n_jobs=-1)(delayed(func)(*arg) for arg in zip(*args))
def run_parallel(func, args):
    return Parallel(n_jobs=os.cpu_count())(delayed(func)(arg) for arg in args)

def load_nifty(directory, example_id, suffix):
    return nib.load(os.path.join(directory, example_id + "-" + suffix + ".nii.gz"))


def load_channels(d, example_id):
    return [load_nifty(d, example_id, suffix) for suffix in ["t2f", "t1n", "t1c", "t2w"]]


def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)


def prepare_nifty(d):
    logger = logging.getLogger(__name__)
    example_id = d.split("/")[-1]
    flair, t1, t1ce, t2 = load_channels(d, example_id)
    affine, header = flair.affine, flair.header
    vol = np.stack([get_data(flair), get_data(t1), get_data(t1ce), get_data(t2)], axis=-1)
    vol = nib.nifti1.Nifti1Image(vol, affine, header=header)
    nib.save(vol, os.path.join(d, example_id + "-stk.nii.gz"))

    if os.path.exists(os.path.join(d, example_id + "-seg.nii.gz")):
        seg = load_nifty(d, example_id, "seg")
        logger.info(f"Segmentation Shape is: {seg.shape}")
        affine, header = seg.affine, seg.header
        vol = get_data(seg, "unit8")
        seg = nib.nifti1.Nifti1Image(vol, affine, header=header)
        nib.save(seg, os.path.join(d, example_id + "-lbl.nii.gz"))


def prepare_dirs(data, train):
    img_path, lbl_path = os.path.join(data, "images_UNN"), os.path.join(data, "labels_UNN")
    call(f"mkdir {img_path}", shell=True)
    if train:
        call(f"mkdir {lbl_path}", shell=True)
    dirs = glob(os.path.join(data, "BraTS*"))
    for d in dirs:
        files = glob(os.path.join(d, "*.nii.gz"))
        for f in files:
            if "t2f" in f or "t1n" in f or "t1c" in f or "t2w" in f or "-seg" in f: 
                continue
            if "-lbl" in f:
                call(f"cp {f} {lbl_path}", shell=True)
            else:
                call(f"cp {f} {img_path}", shell=True)


def prepare_dataset_json(data, train):
    images, labels = glob(os.path.join(data, "images_UNN", "*")), glob(os.path.join(data, "labels_UNN", "*"))
    images = sorted([img.replace(data + "/", "") for img in images])
    labels = sorted([lbl.replace(data + "/", "") for lbl in labels])

    modality = {"0": "t2f", "1": "t1n", "2": "t1c", "3": "t2w"}
    labels_dict = labels_dict = {"0": "background", "1": "NCR", "2": "ED", "3": "ET"}
    if train:
        key = "training"
        data_pairs = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs = [{"image": img} for img in images]

    dataset = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs,
    }

    with open(os.path.join(data, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)


def prepare_dataset(data, train):
    logger = logging.getLogger(__name__)

    print(f"Preparing BraTS21 dataset from: {data}")
    start = time.time()
    run_parallel(prepare_nifty, sorted(glob(os.path.join(data, "BraTS*"))))
    prepare_dirs(data, train)
    prepare_dataset_json(data, train)
    end = time.time()
    print(f"Preparing time: {(end - start):.2f}")


def apply_preprocessing(subject, transform_pipeline, transL):
    logger = logging.getLogger(__name__)
    logger.info("Applying transforms")
    transformed_subject = subject
    for transform_name, transform_func in transform_pipeline.items():
        if transform_name in transL and transform_func is not None:
            logger.info(f"\ntransformation is {transform_name}")
            transformed_subject = transform_func(transformed_subject)
    return transformed_subject

def load_and_transform_images(inputs):
    logger = logging.getLogger(__name__)
    
    pair, data_path = inputs
    
    logger.info("Image-Label pairs are: ", pair)
    logger.info("Mode is: ", pair)

    mode = "training"
    # mode = "test"

    image_path = pair["image"]
    if mode == "training":
        label_path = pair["label"]
        transL = ['checkRAS','CropOrPad','ohe','ZnormFore']
        subject = tio.Subject(
            image=tio.ScalarImage(os.path.join(data_path, image_path)),
            label=tio.LabelMap(os.path.join(data_path, label_path))
        )
    else:
        subject = tio.Subject(
            image=tio.ScalarImage(os.path.join(data_path, image_path)))
        transL = ['checkRAS','ZnormFore']

    transform_pipeline = transforms_preproc(target_shape=True)

    # transL = ['CropOrPad']
    # OPTIONS ARE:
                # 'checkRAS' : to_ras,
                # 'CropOrPad' : crop_pad,
                # 'ohe' : one_hot_enc,
                # 'ZnormFore' : normalise_foreground,
                # 'MaskNorm' : masked,
                # 'Znorm': normalise
                # 'fSSA' : fSSA
    # Apply the preprocessing steps
    transformed_subject = apply_preprocessing(subject, transform_pipeline, transL)
    patient_folder = os.path.basename(image_path).split(".")[0][:-4]

    # Save the transformed images and segmentations to .npy files
    transformed_image = transformed_subject["image"]
    img_npy = transformed_image.numpy()
    logger.info("Image Numpy Shape:",img_npy.shape)
    image_name = os.path.basename(image_path).split(".")[0]
    img_sv_path = os.path.join(data_path, patient_folder, f"{image_name}.npy")
    np.save(img_sv_path, img_npy)
    logger.info("Saved numpy file: ", image_name, ";    to path: ", img_sv_path)

    if mode == "training":
        transformed_label = transformed_subject["label"]
        lbl_npy = transformed_label.numpy()
        logger.info(f"Label Numpy Shape: {lbl_npy.shape}")
        label_name = os.path.basename(label_path).split(".")[0]
        lbl_sv_path = os.path.join(data_path, patient_folder, f"{label_name}.npy")
        logger.info("Saved numpy file: : ", label_name,";    to path: ", lbl_sv_path)  
        np.save(lbl_sv_path, lbl_npy)
    logger.info("DATA PATH : ", data_path, "PATIENT FOLDER :", patient_folder)
     # os.path.splitext(os.path.basename(image_path))[0]
     # os.path.splitext(os.path.basename(label_path))[0]

def preprocess_data(data_dir, args):
    '''
    Function that applies all desired preprocessing steps to an image, as well as to its 
    corresponding ground truth image.

    Returns: preprocessed image (not yet converted to tensor)
    img is still a list of arrays of the 4 modalities from data files
    mask is 3d array
    return img as list of arrays, and mask as before
    '''
    filePaths = json.load(open(os.path.join(data_dir,'dataset.json'), "r"))
    pair = filePaths[args.preproc_set]

    outpath = os.path.join(data_dir, args.data_grp + "_prepoc")
    call(f"mkdir -p {outpath}", shell=True)

    # Load and transform the images and segmentations
    run_parallel(load_and_transform_images, list(zip(pair, itertools.repeat(args.data))))
   
def main():
    current_datetime = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    log_file_name = f"preproc_{current_datetime}.log"
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file_name)

    args = get_main_args()
      
    logging.info("Generating stacked nifti files.")
    startT = time.time()
    logging.info("Loaded all nifti files and saved image data")
    
    if args.preproc_set == "test":
        prepare_dataset(args.data, False)
    else:
        prepare_dataset(args.data, True)

    print("Finished!")
    endT = time.time()
    logging.info(f"Image - label pairs created. Total time taken: {(endT - startT):.2f}")

    startT2 = time.time()
    logging.info("Beginning Preprocessing.")
    preprocess_data(args.data, args)
    end2= time.time()
    logging.info(f"Data Processing complete. Total time taken: {(end2 - startT2):.2f}")
    
if __name__=='__main__':
    main()