# import os
# import nibabel as nib
# import numpy as np
# from PIL import Image

# output_folder = "/home/guest189/data/GLI_jpg_files"  # Create a folder to store the JPEG images
# os.makedirs(output_folder, exist_ok=True)

# nii_files = [file for file in os.listdir('.') if file.endswith('.nii.gz')]

# for file in nii_files:
#     img = nib.load(file)
#     data = img.get_fdata()
#     num_slices = data.shape[-1]  # Number of slices in the last dimension
    
#     for i in range(num_slices):
#         slice_data = data[..., i]
#         slice_data = ((slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))) * 255
#         slice_data = slice_data.astype(np.uint8)
        
#         output_filename = os.path.join(output_folder, f'{os.path.splitext(file)[0]}_{i}.jpg')
#         image = Image.fromarray(slice_data, 'L')
#         image.save(output_filename)
        
#         print(f"Converted slice {i} of {file} to {output_filename}")


import os
import nibabel as nib
import numpy as np
from PIL import Image

output_folder = "/home/guest189/data/GLI_jpg_files"  # Create a folder to store the JPEG images
os.makedirs(output_folder, exist_ok=True)

nii_files = [file for file in os.listdir('.') if file.endswith('.nii.gz')]

for file in nii_files:
    img = nib.load(file)
    data = img.get_fdata()
    
    # Choose the middle slice index
    middle_slice_idx = data.shape[-1] // 2
    
    slice_data = data[..., middle_slice_idx]
    slice_data = ((slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))) * 255
    slice_data = slice_data.astype(np.uint8)
    
    output_filename = os.path.join(output_folder, f'{os.path.splitext(file)[0]}.jpg')
    image = Image.fromarray(slice_data, 'L')
    image.save(output_filename)
    
    print(f"Converted {file} to {output_filename}")