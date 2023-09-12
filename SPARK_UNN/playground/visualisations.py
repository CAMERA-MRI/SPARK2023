import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os

SSA_data_dir='/scratch/guest187/BraTS_Africa_data/SSA_TrainingData/'

# Save SSA images
data = []
modalities = ['t1c', 't1n', 't2f', 't2w']

for folder in os.listdir(SSA_data_dir):
    # check it is one of the original modality files (exclude processed files)
    if not any(i in folder for i in ['stk', 'lbl']) and folder.startswith('B'):
        # get slices
        path = os.path.join(SSA_data_dir, folder)
        imgs = [nib.load(f"{os.path.join(SSA_data_dir, folder)}/{folder}-{m}.nii.gz").get_fdata().astype(np.float32)[:, :, 75] for m in modalities]
        data.append(imgs)

mi = 0
for m in modalities:
    fig, ax = plt.subplots(nrows=8, ncols=6, figsize=(15, 15))
    ax = ax.flatten()
    for a in ax:
        for img in (data):
        # print(img[mi].shape)
            a.imshow(img[mi], cmap='gray')
            a.axis('off')  # the number of images in the grid is 5*5 (25)
    plt.title(modalities[mi])
    plt.tight_layout()  
    plt.savefig('/home/guest187/AlexGit/UNN_BraTS23/reports/' + modalities[mi] + '.png')
    mi+=1 