from glob import glob
import json
import random
import  nibabel as nib
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="generates a json file containing the path to data for each training fold")
parser.add_argument("--data_folder_path", default=None, help="path to the folder where all the training data is stored")
parser.add_argument("--json_file_path", default=None, help="path to the json file where the data fold will be written to")
parser.add_argument("--num_folds", default=5, help="number of k fold cross validation. data in each fold will be used for validation, \
                    the rest of the data is kept for training.")
parser.add_argument("--data_use", default="training", help="is this data used for 'training', 'validation' or 'testing'?\
                    note that testing data is the one without ground truth labels")

def view_data_shape(path_2_img:str):
    r''' Given path to a nifti file, this function will load the contents as numpy array and prints its shape. 
    '''
    img:np.array = nib.load(path_2_img).get_fdata()
    print(img.shape)
    
    return 0

def kfold_data_dict(data_dir:str, num_folds:int, data_use:str, out_json_file:str=None):
    r''' Given a directory of images from BraTS challege, it will map the patient files and stores them in a dictionary
    inputs:
        - data_dir: path to the training data
        - num_folds: number of k fold cross validation 

    output: 
        k_fold_dict: a dictionary holding the following keys:
            fold
            image
            label
            training
    '''
    patient_dir:list = glob(data_dir + "*/")
    random.shuffle(patient_dir)
    # print(patient_dir)

    # Initialize the dictionary holding k fold cross validation
    kfold_dict:dict = {data_use:[]}
    # figure out how many patients are in a fold
    
    num_patient_per_fold:int = (int(len(patient_dir)) // int(num_folds))
    # print(num_patient_per_fold)

    for k in range(num_folds):
        # print(type(k))
        
        for patient in patient_dir[k*num_patient_per_fold: (k+1)*num_patient_per_fold]:
            # print(patient)
            # print()
            temp_dict = {'fold':k}
            temp_dict["image"] = glob(patient+"/*t*")
            if data_use == "training":
                temp_dict["label"] = glob(patient+"/*seg*")[0]
            kfold_dict[data_use].append(temp_dict)
            # print(temp_dict)
            # break
        # break

    if not (out_json_file is None): 
        with open(out_json_file, 'w') as outfile:
            json.dump(kfold_dict, outfile, indent=4)

    return kfold_dict

def main():

    args = parser.parse_args()
    kfold_data_dict(data_dir=args.data_folder_path, num_folds=int(args.num_folds), data_use=args.data_use , out_json_file=args.json_file_path)

    # check the shape of a single image or segmentation file. 
    # for brats 2021 
    # path_2_img = "/scratch/guest183/BraTS_2021_data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/TrainingData/BraTS2021_00000/BraTS2021_00000_seg.nii.gz"
    # for brats 2023 africa 
    # path_2_img = "/scratch/guest183/BraTS_Africa_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00387-000/BraTS-GLI-00387-000-seg.nii.gz"
    # view_data_shape(path_2_img)

    # # On Compute Canada:
    # generate json file for GLI data containing the path to n fold cross validation data 
    # data_dir_gli_training:str = "/scratch/guest183/BraTS_Africa_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
    # kfold_dict_gli_training:dict = kfold_data_dict(data_dir_gli_training, 5, out_json_file="brats23_africa_folds.json")

    # # generate json for GLI validation folder
    # data_dir_GLI_val:str = "/scratch/guest183/BraTS_Africa_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/"
    # json_outdir:str = "./jsons/brats23_gli_test.json"
    # kfold_dict_GLI_testing:dict = kfold_data_dict(data_dir_GLI_val, 1, "testing", json_outdir)

    # Generate json file for sub saharan africa data

    # # On Local PC
    # data_dir_SSA_training = "/home/odcus/Data/BraTS_Africa_data/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData/"
    # json_dict_SSA_training = "/home/odcus/Software/Kilimanjaro_swinUNETR/jsons/brats23_ssa_train.json"
    # kfold_dict_SSA_trainng = kfold_data_dict(data_dir_SSA_training, 5, data_use="training", out_json_file=json_dict_SSA_training)
    

# DO NOT DELETE
if __name__ == "__main__":
    main()
