import os 
import random
import shutil

# after unzipping the “RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021.zip” from data_scratch_dir 
# reanming the folder to just data 
data_scratch_dir = "/scratch/guest189/BraTS2023_data/BraTS_2021_data/"
train_data_dir = "RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"

old_folder_name = data_scratch_dir + train_data_dir
new_folder_name = data_scratch_dir + "data"

# Rename the folder
os.rename(old_folder_name, new_folder_name)

# splitting the subfolders inside the new_folder_name '/scratch/guest189/BraTS_2021_data/data' into 80% train and 20% validation set 


source_folder = new_folder_name
train_folder = '/scratch/guest189/BraTS2023_data/BraTS_2021_data/BraTS2021_train'
val_folder = '/scratch/guest189/BraTS2023_data/BraTS_2021_data/BraTS2021_val'

def create_train_val_folder(train_folder, val_folder):

    # Create the  train_folder if it doesn't exist
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    # Create the val_folder if it doesn't exist
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    # Get a list of subfolders in the source folder
    subfolders = [name for name in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, name))]

    # Set the random seed for reproducibility
    random.seed(42)

    # Shuffle the subfolders randomly
    random.shuffle(subfolders)

    # Define the number of subfolders to allocate for validation
    num_val_subfolders = int(len(subfolders) * 0.2)  # 20% for validation

    # Move the subfolders to the respective train or val folder
    for i, subfolder in enumerate(subfolders):
        source_path = os.path.join(source_folder, subfolder)
        if i < num_val_subfolders:
            destination_path = os.path.join(val_folder, subfolder)
        else:
            destination_path = os.path.join(train_folder, subfolder)
        # shutil.move(source_path, destination_path)
        shutil.copytree(source_path, destination_path)

create_train_val_folder(train_folder, val_folder)
