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

import nibabel as nib
import numpy as np
import torch
import torchio as tio


from utils.utils import get_main_args
from utils.utils import extract_imagedata
from data_transforms import transforms_preproc

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)



def run_parallel(func, args):
    return Parallel(n_jobs=os.cpu_count())(delayed(func)(arg) for arg in args)

def data_preparation(data_dir, args=get_main_args()):
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
    # Step 1: Initialisation
    data_dir = data_dir # path for each subject folder in the set
    modalities = args.modal
    subj_dirs, subj_dir_pths = [],[]
    # store images to load (paths)
    img_pth, seg_pth = [],[]
    file_ext_dict_prep = {
        **{f"-{m}.nii.gz": img_pth for m in modalities},
        "-seg.nii.gz": seg_pth}
                
    #Loop through main data folder to generate lists of paths
    logging.info(f"Generating dataset paths from data folder: {data_dir}")
    for root, dirs, files in os.walk(data_dir):
        for directory in sorted(glob(os.path.join(dirs, "BraTS*"))):
            if not "BraTS-" in directory:
                continue
            else:
                subj_dirs.append(str(directory))
                subj_dir_pths.append(os.path.join(data_dir,directory))
            for file in files:
                file_pth = os.path.join(root, file)
                if os.path.isfile(file_pth) and args.task=='data_prep':
                    for ext, list_to_append in file_ext_dict_prep.items():
                        if file.endswith(ext):
                            #logging.info(file_pth)
                            list_to_append.append(file_pth)
    
    logging.info(f"Saving path lists to file: {args.preproc_set}_paths.json")
    with open(os.path.join(data_dir, f'{args.preproc_set}_paths.json'), 'w') as file:
        json.dump(file_ext_dict_prep, file)

    # Step 2: Stack modalities into 1 nii file, and extract header information
    logging.info("Preparing stacked nifty files")
    img_shapes = {}
    res = {}
    img_modality = []
    ext_dict_modal = {**{f"-{m}.nii.gz": img_modality for m in modalities}}
    # store paths
    proc_imgs, proc_lbls = [],[]
    file_ext_dict_prep2 = {
        "-stk.nii.gz": proc_imgs,
        "-lbl.nii.gz": proc_lbls
    }
    
    for sub_dir in sorted(subj_dir_pths, key=lambda x: x.lower()):
        if not "BraTS-" in sub_dir:
            continue
        subj_id = os.path.basename(sub_dir)
        logging.info(f"Working on subj: {subj_id}")
        
    #Load nifti file for each scanning sequence
        logging.info("Loading and stacking modalities")
        img_paths = [s for s in img_pth if subj_id in s]
        loaded_modalities = [nib.load(path) for path in img_paths]
        t1n, t1c, t2w, t2f = loaded_modalities
        img_modality.extend([t1n, t1c, t2w, t2f]) 
        affine, header = t2f.affine, t2f.header
        
        res[f'{subj_id}_RES']=header.get_zooms()
    
    #Stack all into one nifti file
        imgs = np.stack([extract_imagedata(modality) for modality in loaded_modalities], axis=-1)
        shapes = {modality: imgs[..., i].shape for i, modality in enumerate(modalities)}
        img_shapes[f'{subj_id}'] = shapes
        logging.info(f"Image shapes: {img_shapes}")
        imgs = nib.nifti1.Nifti1Image(imgs, affine, header=header)
        
        fPath = os.path.join(sub_dir, subj_id + "-stk.nii.gz")
        proc_imgs.append(fPath)       

        # if os.path.exists(fPath):
        #     # Delete the existing file
        #     os.remove(fPath)
        # nib.save(imgs, fPath)
        del imgs
    # Step 3: Load and save seg
        logging.info("Loading and saving segmentation")
        seg = nib.load(os.path.join(sub_dir, subj_id + "-seg.nii.gz"))
        seg_affine, seg_header = seg.affine, seg.header
        seg = extract_imagedata(seg, "unit8")
        #seg[vol == 4] = 3 --> not sure what this does yet
        seg = nib.nifti1.Nifti1Image(seg, seg_affine, header=seg_header)
        logging.info(f"Seg Shape {seg.shape}")

        fPath2 = os.path.join(sub_dir, subj_id + "-lbl.nii.gz")
        proc_lbls.append(fPath2)

        # if os.path.exists(fPath):
        #     # Delete the existing file
        #     os.remove(fPath)
        nib.save(seg, fPath2)
        del seg
                       
    # save a few bits of info into a json 
    with open(os.path.join(args.data, f'{args.preproc_set}_paths.json'), 'w') as file:
        json.dump(file_ext_dict_prep2, file)
    
    logging.info(f"Saving subject folder paths and list of IDs. Total subjects is: {len(subj_dirs)}")    
    subj_info = {
        "nSubjs" : len(subj_dirs),
        "subjIDs" : subj_dirs,
        "subj_dirs" : subj_dir_pths
    }
    with open(os.path.join(args.data, "subj_info.json"), "w") as file:
        json.dump(subj_info,file)
    locals().clear()

def file_prep(data_dir, args):
    """ 
    This an extra function to save a copy of the image data extracted from each volume.
    data_loader and trainer do not require these data as they are stored in the original subject folders as well.

    Creates a json file with
        A dictionary of dummy coding for seg labels as provided by BraTS
            "labels" : {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"}
        A dictionary of dummy coding for each modality
            "modality": {"0": "FLAIR", "1": "T1", "2": "T1CE", "3": "T2"}
        A dictionary of dictionaries containing the image-label path pairs
            "training": [{"image": "images/subjIDxxx.nii.gz", "label": "labels/subjIDxxx_seg.nii.gz"}
    """
    
    filePaths = json.load(open(os.path.join(data_dir,f'{args.preproc_set}_paths.json'), "r"))
    stk = sorted(filePaths["-stk.nii.gz"], key=lambda x: x.lower(), reverse=True)
    lbl = sorted(filePaths["-lbl.nii.gz"], key=lambda x: x.lower(), reverse=True)

    subjInfo = json.load(open(os.path.join(data_dir,'subj_info.json'), "r"))
    subj_dirs = subjInfo["subj_dirs"]
    subj_id = subjInfo["subjIDs"]
    
    stk_path, lbls_path = os.path.join(data_dir, f"images_orig-{args.data_grp}"), os.path.join(data_dir, f"labels_orig-{args.data_grp}")
    call(f"mkdir -p {stk_path}", shell=True)
    if args.preproc_set != "test":
        call(f"mkdir -p {lbls_path}", shell=True)
    
    imagesF, labelsF = [], []

    for dir in sorted(subj_dirs, key=lambda x: x.lower(), reverse=True):
        if not "BraTS-" in dir:
            break
        id_check = os.path.basename(dir)
        for i in range(len(subj_id)):
            if id_check == os.path.dirname(lbl[i]):
                lbl_file = os.path.basename(lbl[i]) 
                labelsF.append(os.path.join(dir, lbl_file))
                call(f"cp {lbl[i]} {lbls_path}", shell=True)
            if id_check == os.path.dirname(stk[i]):            
                stk_file = os.path.basename(stk[i]) 
                imagesF.append(os.path.join(dir, stk_file))
                call(f"cp {stk[i]} {stk_path}", shell=True)

    if args.preproc_set == "training":
        key = "training"
        data_pairs = [{"image": imgF, "label": lblF} for (imgF, lblF) in zip(imagesF, labelsF)]
    else:
        key = "test"
        data_pairs = [{"image": imgF} for imgF in imagesF]

    modality = {"0": "t1n", "1": "t1c", "2": "t2w", "3": "t2f"}
    labels_dict = {"0": "background", "1": "NCR", "2": "ED", "3": "ET"}

    # sAve some json files for dataloading
    dataset = {
        "labels": labels_dict,
        "modality": modality,
        "subjIDs" : subj_id,
        key: data_pairs}
    with open(os.path.join(data_dir, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)
    
    locals().clear()

def apply_preprocessing(subject, transform_pipeline, transL):
    transformed_subject = subject
    for transform_name, transform_func in transform_pipeline.items():
        if transform_name in transL and transform_func is not None:
            transformed_subject = transform_func(transformed_subject)
    return transformed_subject

def load_and_transform_images(pair):
    images = []
    labels = []
    for item in pair:
        image_path = item["image"]
        label_path = item["label"]
        
        # Load the image and label using TorchIO
        subject = tio.Subject(
            image=tio.ScalarImage(image_path),
            label=tio.LabelMap(label_path)
        )
        transform_pipeline = transforms_preproc()
        transL = ['checkRAS','ohe','ZnormFore']
        
        # Apply the preprocessing steps
        transformed_subject = apply_preprocessing(subject, transform_pipeline, transL)
        
        transformed_image = transformed_subject["image"]
        transformed_label = transformed_subject["label"]
        images.append(transformed_image)
        labels.append(transformed_label)

    return images, labels

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
    pair = filePaths["training"]

    outpath = os.path.join(data_dir, args.data_grp + "_prepoc")
    call(f"mkdir -p {outpath}", shell=True)

    # transforms = []
    
    # for code, trans in transform_pipeline.items():
    #     if code in transList:
    #         transforms.append(trans)
    # transform = tio.Compose(transforms)
    
   # Load and transform the images and segmentations
    transformed_images, transformed_labels = run_parallel(load_and_transform_images, pair)

    # Save the transformed images and segmentations to .npy files
    for i, (image_path, label_path) in enumerate(zip([item["image"] for item in pair], [item["label"] for item in pair])):
        img_npy = transformed_images[i].numpy()
        seg_npy = transformed_labels[i].numpy()
        logging.info(f"Image Numpy Shape: {img_npy.shape}")
        logging.info(f"Seg Numpy Shape: {seg_npy.shape}")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        label_name = os.path.splitext(os.path.basename(label_path))[0]
        np.save(os.path.join(args.data, image_name[:-4], f"{image_name}.npy"), img_npy)
        np.save(os.path.join(args.data, label_name[:-4], f"{label_name}.npy"), seg_npy)

def main():
    
    current_datetime = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    log_file_name = f"app_{current_datetime}.log"
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file_name)

    # startT = time.time()
    args = get_main_args()
    data_dir = args.data
    
    logging.info("Generating stacked nifti files.")
    run_parallel(data_preparation, data_dir)
    logging.info("Loaded all nifti files and saved image data")

    logging.info("Saving a copy to images and labels folders")
    file_prep(data_dir, args)
    endT = time.time()
    
    logging.info(f"Image - label pairs created. Total time taken: {(endT - startT):.2f}")

    startT2 = time.time()
    logging.info("Beginning Preprocessing.")
    
    preprocess_data(data_dir, args)
    end2= time.time()
    
    logging.info(f"Data Processing complete. Total time taken: {(end2 - startT2):.2f}")

if __name__=='__main__':
    main()


#['checkRAS', 'CropOrPad', 'ohe' , 'Znorm']
# OPTIONS ARE:
    # 'checkRAS' : to_ras,
    # 'CropOrPad' : crop_pad,
    # 'ohe' : one_hot_enc,
    # 'ZnormFore' : normalise_foreground,
    # 'MaskNorm' : masked,
    # 'Znorm': normalise
    # 'fSSA' : fSSA