import os
import pickle
import glob
import json
import logging
from subprocess import call

from data_class import MRIDataset
from data_transforms import define_transforms
from utils.utils import get_main_args

import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split


def main():
    """
    This main function is not called during training.

    Use for testing that dataloaders, and data class module work correctly
    """
    args = get_main_args()

    ## Alex: testing from terminal
    data_dir = '/scratch/guest187/Data/train_all'
    
    data_folders = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if not (file.endswith(".json") or file == 'images' or file == 'labels' or file == 'ATr_prepoc')]

    print(data_folders)
    
    batch_size = 8

    ## Code to save config file
    pickle.dump(
        {
            "patch_size": [128, 128, 128],
            "spacings": [1.0, 1.0, 1.0],
            "n_class": 4,
            "in_channels": 4,
        },
        open(os.path.join('/scratch/guest187/Data/train_all', "config.pkl"), "wb"),
    )

    dataloaders = load_data(data_dir, batch_size, args)
    print(dataloaders)
    
    training_set = dataloaders['train']
    
    for img, label in training_set:
        print(f"Image shape: {img.shape}")
        print(f"Label shape: {label.shape}")
    

def load_data(args, data_transforms):

    '''
    This function is called during training after define_transforms(n_channels)

    It takes as input
        args: argparsers from the utils script 
            args.seed
            args.data_used: 'all', 'GLI', 'SSA'
        data_transforms: a dictionary of transformations to apply to the data during training

    Returns dataloaders ready to be fed into model
    '''
    logger = logging.getLogger(__name__)

    # Set a seed for reproducibility if you want the same split - optional
    if args.seed != None:
        seed=args.seed
        logger.info(f"Seed set to {seed}.")
    else:
        seed=None
        logger.info("No seed has been set")
    
    fakeSSA = None

    # Locate data based on which dataset is being used
    # fakeSSA transforms are applied to GLI data to worse their image quality
    logger.info(f"Data used is {args.data_used}.")

    if args.data_used == 'ALL':
        data_folders = glob.glob(os.path.join(args.data, "BraTS*"))
    elif args.data_used == "GLI":
        data_folders = [folder for folder in os.listdir(args.data) if 'GLI' in folder]
    elif args.data_used == 'SSA':
        data_folders = [folder for folder in os.listdir(args.data) if 'SSA' in folder]
   
    ###### We now get data transforms before calling load_data
        # Get data transforms
        # data_transforms = define_transforms(n_channels)

    
    # image_datasets = {
    #     'train': MRIDataset(args.data,train_files, transform=data_transforms['train'], SSAtransform=data_transforms['fakeSSA']),
    #     'val': MRIDataset(args.data,val_files, transform=data_transforms['val']),
    #     # 'test': MRIDataset(args, test_files, transform=data_transforms['test'])
    # }
    if args.exec_mode == "train":
        # Split data files
        train_files, val_files = split_data(data_folders, seed) 
        logger.info(f"Number of training files: {len(train_files)}\nNumber of validation files: {len(val_files)}")
        # if args.data_used != "SSA" and args.augs is not None:
        #     fakeSSA = data_transforms['fakeSSA'][str(args.augs)]
        fakeSSA = data_transforms['fakeSSA']
        logger.info(f"Fake SSA transforms call is {fakeSSA}.")

        image_datasets = {
            'train': MRIDataset(args.data, train_files, transform=data_transforms['train'], SSAtransform=fakeSSA),
            'val': MRIDataset(args.data, val_files, transform=data_transforms['val']),
        }
        # Create dataloaders
        # can set num_workers for running sub-processes    
        dataloaders = {
            'train': data_utils.DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, drop_last=True),
            'val': data_utils.DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=True)
            # 'test': data_utils.DataLoader(image_datasets['test'], batch_size=args.val_batch_size, shuffle=True)
        }

        # Save data split
        splitData = {
            'subjsTr' : train_files,
            'subjsVal' : val_files,
            # 'subjsTest' : test_files    
        }
        with open(args.data + str(args.data_used) + ".json", "w") as file:
            json.dump(splitData, file)

    elif args.exec_mode == "predict":
        val_files = [os.path.join(args.data, file) for file in os.listdir(args.data) if (file.startswith("BraTS-"))]
        image_datasets = {'val': MRIDataset(args.data, val_files, transform=data_transforms['val'])}
        
        dataloaders = {'val': data_utils.DataLoader(image_datasets['val'], batch_size=1, shuffle=False)}

    return dataloaders

def split_data(data_folders, seed):
    '''
    Function to split dataset into train/val/test splits, given all avilable data.
    Input:
        list of paths to numpy files
    Returns:
        lists for each train and val/test sets, where each list contains the file names to be used in the set
    '''
    #-----------------------------
    # originally we split as 3: train-test-val train (70), val (15), test (15):
        # train_files, test_files = train_test_split(data_folders, test_size=0.7, random_state=seed)
        # val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=seed)

    #-----------------------------
    # training loop split is train-val (70-30)
    train_files, val_files = train_test_split(data_folders, test_size=0.3, random_state=seed)

    # ??? validation/testing???

    return train_files, val_files

if __name__=='__main__':
    main()