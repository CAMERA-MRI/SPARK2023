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
# from joblib import Parallel, delayed

import nibabel as nib
import numpy as np
import torch

import torchio as tio
import torchvision.transforms as transforms


import utils.utils
from utils.utils import get_main_args
from utils.utils import extract_imagedata
from data_transforms import transforms_preproc

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)
    
# def run_parallel(func, *args):
#     return Parallel(n_jobs=-1)(delayed(func)(*arg) for arg in zip(*args))

def prepare_paths(args):

    data_dir = args.data  # path for each subject folder in the set
    modalities = args.modal
    task = args.task
    
    subj_dirs, subj_dir_pths = [],[]
    # store images to load (paths)
    img_pth, seg_pth = [],[]
   
    for root, dirs, files in os.walk(data_dir):
        for directory in sorted(dirs, key=lambda x: x.lower(), reverse=True):
            if not "BraTS-" in directory:
                break
            else:
                subj_dirs.append(str(directory))
                subj_dir_pths.append(os.path.join(root,directory))
                #subjIDls.append(subj_dirs)
        for file in files:
            file_pth = os.path.join(root, file)
            if os.path.isfile(file_pth) and task=='data_prep':
                if any(string in file_pth for string in modalities):
                    img_pth.append(file_pth)
                elif "-seg.nii.gz" in file_pth:
                    seg_pth.append(file_pth)
    file_ext_dict_prep = {
        "-m.nii.gz": img_pth,
        "-seg.nii.gz": seg_pth}
    with open(os.path.join(data_dir, f'{args.preproc_set}_OrigPaths.json'), 'w') as file:
        json.dump(file_ext_dict_prep, file)
    logging.info(f"Saving subject folder paths and list of IDs. Total subjects is: {len(subj_dirs)}")    
    
    subj_info = {
        "nSubjs" : len(subj_dirs),
        "subjIDs" : subj_dirs,
        "subj_dirs" : subj_dir_pths
    }
    with open(os.path.join(data_dir, "subj_info.json"), "w") as file:
        json.dump(subj_info,file)
           
    return img_pth, seg_pth, subj_dir_pths

def prepare_nifty(img_pth, seg_pth, subj_dir_pths):
    """ 
    This is the main data prepartion function. 
    It extracts the the image data from each volume and then stacks all modalities into one file.
    It then applies standard image preprocessing such as one hot encoding, realignment to RAS+ Z normalisation
    data_loader and trainer will work with these files.
    Input:
        dataset class
        args
        # OLD: 
            path to directory containing folders of subject IDs
            list of modalities
    Output:
        JSON files:]
            subj_info == subject IDs & dir paths
            image_info == shape & resolution data per subject per modality
            dataset == modality keys, segmentation keys, image-label pairs per subj
        NifTI files:.
            subjIDxxx-stk.nii.gz == stacked nifti img data 
            subjIDxxx-lbl.nii.gz == seg mask img data

    """
    
    modalities = ["t1c", "t1n", "t2w", "t2f"]
    img_shapes = {}
    res = {}
    
    for sub_dir in sorted(subj_dir_pths, key=lambda x: x.lower(), reverse=False):
        subj_id = os.path.basename(sub_dir)
        logging.info(f"Working on subj: {subj_id}")
           
    #Load nifti file for each scanning sequence
        logging.info("Loading and stacking modalities")
        img_paths = [s for s in img_pth if subj_id in s]
        logging.info(f"Image paths are: {img_paths}")
        loaded_modalities = [nib.load(path) for path in img_paths]
        t1n, t1c, t2w, t2f = loaded_modalities
        affine, header = t2f.affine, t2f.header
        res[f'{subj_id}_RES']=header.get_zooms()
    
        #Stack all into one nifti file
        imgs = np.stack([extract_imagedata(modality) for modality in loaded_modalities], axis=-1)
        shapes = {modality: imgs[..., i].shape for i, modality in enumerate(modalities)}
        img_shapes[f'{subj_id}'] = shapes
        logging.info(f"Image shapes: {img_shapes[{subj_id}]}")
        imgs = nib.nifti1.Nifti1Image(imgs, affine, header=header)
        nib.save(imgs, os.path.join(sub_dir, subj_id + "-stk.nii.gz"))
    #Load and save segmentation volume
        seg_path = [s for s in seg_pth if subj_id in s]
        seg = nib.load(seg_path)
        seg_affine, seg_header = seg.affine, seg.header
        seg = extract_imagedata(seg, "unit8")
        #seg[vol == 4] = 3 --> not sure what this does yet
        seg = nib.nifti1.Nifti1Image(seg, seg_affine, header=seg_header)
        logging.info(f"Seg Shape {seg.shape}")
        nib.save(seg, os.path.join(sub_dir, subj_id + "-lbl.nii.gz"))
        del imgs
        del seg

def dirs_prep(args):
    """ 
    This an extra function to save a copy of the image data extracted from each volume.
    data_loader and trainer do not require these data as they are stored in the original subject folders as well

    Creates a json file with
        A dictionary of dummy coding for seg labels as provided by BraTS
            "labels" : {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"}
        A dictionary of dummy coding for each modality
            "modality": {"0": "FLAIR", "1": "T1", "2": "T1CE", "3": "T2"}
        A dictionary of dictionaries containing the image-label path pairs
            "training": [{"image": "images/subjIDxxx.nii.gz", "label": "labels/subjIDxxx_seg.nii.gz"}
    """
    data_dir = args.data
    stk_path, lbls_path = os.path.join(data_dir, f"images_orig-{args.data_grp}"), os.path.join(data_dir, f"labels_orig-{args.data_grp}")
    call(f"mkdir -p {stk_path}", shell=True)
    if args.preproc_set != "test":
        call(f"mkdir -p {lbls_path}", shell=True)
        
    imagesF, labelsF = [], []
    dirs = glob(os.path.join(data_dir, "BraTS*"))
    for d in dirs:
        if "-" in d.split("/")[-1]:
            files = glob(os.path.join(d, "*.nii.gz"))
            for f in files:
                if "t2f" in f or "t1n" in f or "t1c" in f or "t2w" in f:
                    continue
                if "-lbl" in f:
                    labelsF.append(os.path.join(d, f))
                    call(f"cp {f} {lbls_path}", shell=True)
                else:
                    imagesF.append(os.path.join(d, f))
                    call(f"cp {f} {stk_path}", shell=True)
    
    if args.preproc_set != "test":
        key = "training"
        data_pairs = [{"image": imgF, "label": lblF} for (imgF, lblF) in zip(imagesF, labelsF)]
    else:
        key = "test"
        data_pairs = [{"image": imgF} for imgF in imagesF]
    modality = {"0": "t1n", "1": "t1c", "2": "t2w", "3": "t2f"}
    labels_dict = {"0": "background", "1": "NCR", "2": "ED", "3": "ET"}

    # **********These path pairs are not needed for data_loader or training--> this is for incase it is needed
    images, labels = glob(os.path.join(stk_path, "*")), glob(os.path.join(lbls_path, "*"))
    images = sorted([img.replace(data_dir + "/", "") for img in images])
    labels = sorted([lbl.replace(data_dir + "/", "") for lbl in labels])
    if args.preproc_set != "test":
        key = "training"
        data_pairs_fold = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs_fold = [{"image": img} for img in images]

    # sAve some json files for dataloading
    dataset = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs}
    with open(os.path.join(data_dir, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)

    datasetFold = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs_fold}
    with open(os.path.join(data_dir, "datasetFold.json"), "w") as outfile:
        json.dump(datasetFold, outfile)


def preprocess_data(args,transList):
    '''
    Function that applies all desired preprocessing steps to an image, as well as to its 
    corresponding ground truth image.

    Returns: preprocessed image (not yet converted to tensor)
        # img is still a list of arrays of the 4 modalities from data files
    mask is 3d array

    return img as list of arrays, and mask as before
    '''
    data_dir = args.data
   
    outpath = os.path.join(data_dir, args.data_grp + "_prepoc")
    call(f"mkdir -p {outpath}", shell=True)

    # Define the list of helper functions for the transformation pipeline
    transform_pipeline = transforms_preproc(args.target_shape)[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = glob(os.path.join(data_dir, "BraTS*"))
    for d in dirs:
        files = glob(os.path.join(d, "*.nii.gz"))
        for f in files:
            if "-stk.nii.gz" not in f and "-lbl.nii.gz" not in f:
                continue
            elif "-stk.nii.gz" in f:
                proc_img = nib.load(f)
                proc_img = extract_imagedata(proc_img)
                proc_img_t = (torch.from_numpy(proc_img)).to(device)
                for code, trans in transform_pipeline.items():
                    if code in transList:
                        proc_img_t = trans(proc_img_t)
                np.save(os.path.join(os.path.dirname(f), str(d) + "-stk.npy"), proc_img_t)
            elif "-lbl.nii.gz" in f:
                proc_lbl = nib.load(f)
                proc_lbl = extract_imagedata(proc_lbl)
                proc_lbl_t = (torch.from_numpy(proc_lbl)).to(device)
                proc_lbl_t = torch.unsqueeze(proc_lbl_t, axis=0)
                for code, trans in transform_pipeline.items():
                    if code in transList:
                        proc_lbl_t = trans(proc_lbl_t)
                np.save(os.path.join(os.path.dirname(f), str(d) + "-lbl.npy"), proc_img_t)

def main():
    logging.basicConfig(filename='04-07_data_prep_22h40.log', filemode='w', level=logging.DEBUG)
    args = get_main_args()
    utils.utils.set_cuda_devices(args)
      
    logging.info("Generating stacked nifti files.")
    startT = time.time()
    img_pth, seg_pth, subj_dir_pths = prepare_paths(args)
    
    prepare_nifty(img_pth, seg_pth, subj_dir_pths)
    logging.info("Loaded all nifti files and saved image data")
    
    logging.info("Saving a copy to images and labels folders")
    dirs_prep(args)
    endT = time.time()
    logging.info(f"Image - label pairs created. Total time taken: {(endT - startT):.2f}")

    logging.info("Beginning Preprocessing.")
    startT2 = time.time()
    transL = ['checkRAS', 'CropOrPad', 'Znorm']
        # transform_pipeline = {
        # 'checkRAS' : to_ras,
        # 'CropOrPad' : crop_pad,
        # 'ohe' : one_hot_enc,
        # 'ZnormFore' : normalise_foreground,
        # 'MaskNorm' : masked,
        # 'Znorm': normalise
    # procArgs = (args, transL)
    preprocess_data(args,transL)
    
    end2= time.time()
    logging.info(f"Data Processing complete. Total time taken: {(end2 - startT2):.2f}")
    
if __name__=='__main__':
    main()