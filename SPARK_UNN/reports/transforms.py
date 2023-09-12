import os
import torch
import torchio as tio
import matplotlib.pyplot as plt

# Set the main data directory path
data_dir = '/scratch/guest187/BraTS2023_OriginalData/TrainingData_release/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2'
output_dir = '/home/guest187/GitRepo_Brats23/UNN_BraTS23/reports'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the list of transformations to apply
transformations = [
    tio.ToCanonical(),
    tio.CropOrPad((192, 224, 160), mask_name="seg"),
    tio.CropOrPad((192, 224, 160)),
    tio.CropOrPad((192, 192, 124)),
    tio.CropOrPad(mask_name="seg"),
    tio.RandomFlip(axes=(0, 1, 2), p=0.3),
    tio.Resample((1.2, 1.2, 6)),
    tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1, 6)),
    tio.RandomBlur(std=(0.5, 1.5)),
    tio.RandomNoise(mean=0, std=(0, 0.33)),
    tio.RandomMotion(num_transforms=3, image_interpolation='nearest'),
    tio.RandomBiasField(coefficients=1),
    tio.RandomGhosting(intensity=1.5)
]

# Iterate through the subject folders
subject_dirs = sorted([os.path.join(data_dir, subject_dir) for subject_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subject_dir))])
for subject_dir in subject_dirs:
    subject_id = os.path.basename(subject_dir)
    print(subject_dir)
    # Create a list to hold the subject's images and labels
    images = []
    labels = []

    # Load each scan modality and segmentation
    for modality in ['t1n', 't1c', 't2w', 't2f']:
        image_path = os.path.join(subject_dir, f'{subject_id}-{modality}.nii.gz')
        image = tio.ScalarImage(image_path)
        images.append(image)

    label_path = os.path.join(subject_dir, f'{subject_id}-seg.nii.gz')
    label = tio.LabelMap(label_path)
    labels.append(label)

    # Create the subject using the images and labels
    subject = tio.Subject(
        t1n=images[0],
        t1c=images[1],
        t2w=images[2],
        t2f=images[3],
        seg=labels[0]
    )

    # Create the dataset with the subject
    dataset = tio.SubjectsDataset([subject])

    # Apply transformations and save the resulting figures
    transformed_subjects = []
    for current_transformation in transformations:
        for current_subject in dataset:
            transformed_subject = current_transformation(current_subject)
            transformed_subjects.append(transformed_subject)

transformed_dataset = tio.SubjectsDataset(transformed_subjects)

# Create a figure for all views
fig, axes = plt.subplots(len(subject_dirs), 4, figsize=(12, 3 * len(subject_dirs)))

# Iterate through the transformed subjects and plot the images
for subject_index, transformed_subject in enumerate(transformed_dataset):
    # Get the transformed images and labels
    transformed_images = [transformed_subject['t1n'], transformed_subject['t1c'], transformed_subject['t2w'], transformed_subject['t2f']]
    transformed_label = transformed_subject['seg']

# Iterate through the views and plot the images
    for j, view in enumerate(['axial', 'coronal', 'sagittal']):
        ax = axes[subject_index, j]
        # Convert view to an integer if necessary
        if isinstance(view, str):
            view = {'axial': 0, 'coronal': 1, 'sagittal': 2}[view]
        ax.imshow(transformed_images[j].data.squeeze().numpy()[:,:,view])
        ax.axis('off')
        ax.set_title(f'{modality.upper()} {view}')

        # Plot the segmentation label for the current subject
        ax = axes[subject_index, 3]
        ax.imshow(transformed_label.data.squeeze().numpy()[:,:,view])
        ax.axis('off')
        ax.set_title('Segmentation')

# Save the figure as an EPS file
output_filename = f'all_subjects_{view}_{transformation.name}.eps'
output_path = os.path.join(output_dir, output_filename)
plt.savefig(output_path, format='eps')

plt.close(fig)