# import os
# import nibabel as nib
# import numpy as np
# import gzip

# def nii_to_npy_gz(input_file):
#     # Load the NIfTI file
#     nii_data = nib.load(input_file)
#     # Get the raw image data
#     image_data = nii_data.get_fdata()
#     # Save the data to a temporary .npy file
#     npy_file = input_file.replace('.nii.gz', '.npy')
#     np.save(npy_file, image_data)
#     # Compress the .npy file to .npy.gz
#     with open(npy_file, 'rb') as f_in:
#         with gzip.open(npy_file.replace('.npy', '.npy.gz'), 'wb') as f_out:
#             f_out.writelines(f_in)
#     # Remove the temporary .npy file
#     os.remove(npy_file)

# # Get the list of files in the current directory
# files_in_directory = os.listdir()
# # Filter for NIfTI files with .nii.gz extension
# nifti_files = [file for file in files_in_directory if file.endswith('.nii.gz')]
# # Process each NIfTI file
# for nifti_file in nifti_files:
#     print(f"Converting {nifti_file} to NumPy compressed format...")
#     nii_to_npy_gz(nifti_file)

# print("Conversion completed for all NIfTI files.")

# #########################################################################3


# import os
# import numpy as np
# import nibabel as nib
# from PIL import Image

# def nii_to_jpeg(input_file):
#     # Load the NIfTI file
#     nii_file = nib.load(input_file)
#     data = nii_file.get_fdata()
    
#     # Choose a slice index (e.g., middle slice along the Z-axis)
#     z_slice_index = data.shape[2] // 2
#     slice_data = data[:, :, z_slice_index]

#     # Normalize the slice data to 0-255 (assuming the data has appropriate scaling)
#     normalized_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
#     normalized_data = normalized_data.astype(np.uint8)

#     # Convert to a PIL image
#     image = Image.fromarray(normalized_data)

#     # Convert 'RGBA' to 'RGB' mode if the image has an alpha channel
#     if image.mode == 'RGBA':
#         image = image.convert('RGB')

#     # Save as JPEG with the same name as the input file but with the .jpg extension
#     output_file = input_file.replace('.nii.gz', '.jpg')
#     image.save(output_file)

# # Get the list of files in the current directory
# files_in_directory = os.listdir()

# # Filter for NIfTI files with .nii.gz extension
# nifti_files = [file for file in files_in_directory if file.endswith('.nii.gz')]

# # Process each NIfTI file
# for nifti_file in nifti_files:
#     print(f"Converting {nifti_file} to JPEG...")
#     nii_to_jpeg(nifti_file)

# print("Conversion completed for all NIfTI files.")


# Back to Nifti
import os
import numpy as np
import nibabel as nib
from PIL import Image

def resize_png_to_nii(png_path, output_nii_path, target_shape=(240, 240, 155)):
    # Load the PNG image using PIL
    png_image = Image.open(png_path)

    # Resize the PNG image to the target shape
    resized_image = png_image.resize((target_shape[0], target_shape[1]), Image.LANCZOS)

    # Convert the resized PNG image to a numpy array
    resized_array = np.array(resized_image)

    # Create an empty 3D numpy array with the target shape
    target_array = np.zeros(target_shape, dtype=np.uint8)

    # Place the resized PNG array into the target array at the appropriate position
    z_start = (target_shape[2] - resized_array.shape[2]) // 2
    target_array[:, :, z_start:z_start + resized_array.shape[2]] = resized_array

    # Save the target array as a .nii.gz file using nibabel
    nib_image = nib.Nifti1Image(target_array, affine=np.eye(4))
    nib.save(nib_image, output_nii_path)

    # Get the current directory path
current_directory = os.getcwd()

# List all files in the current directory
files_in_directory = os.listdir(current_directory)

# Filter PNG files
png_files = [file for file in files_in_directory if file.lower().endswith(".png")]

# Process each PNG file
for png_file in png_files:
    # Create the output file name for the corresponding .nii.gz file
    nii_file_name = png_file.replace(".png", ".nii.gz")
    nii_file_path = os.path.join(current_directory, nii_file_name)

    # Call the resize_png_to_nii function to convert the PNG to .nii.gz
    png_file_path = os.path.join(current_directory, png_file)
    resize_png_to_nii(png_file_path, nii_file_path)

print("Conversion complete!")