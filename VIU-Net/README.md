# Optimized Brain Tumor Segmentation for resource constrained settings: VGG-Infused U-Net Approach

## Overview

VIU-Net is a brain tumor segmentation model designed to address the challenges of semantic segmentation in low-resource settings. By simplifying the complex 3D multi-label segmentation problem into 2D single-label tasks, the model effectively reduces computational demands. The architecture integrates a pre-trained VGG19 model with a U-Net, enhancing feature extraction while maintaining efficiency.

## Authors

- **Mizanu Zelalem Degu**  
  Faculty of Computing and Informatics, Jimma University, Jimma, Ethiopia  
  ORCID: [0000-0002-8619-5747](https://orcid.org/0000-0002-8619-5747)

- **Confidence Raymond**  
  Medical Artificial Intelligence Laboratory (MAI Lab), Lagos, Nigeria  
  Lawson Health Research Institute, London, Ontario, Canada  
  ORCID: [0000-0003-3927-9697](https://orcid.org/0000-0003-3927-9697)

- **Dong Zhang**  
  Department of Electrical and Computer Engineering, University of British Columbia, Vancouver, Canada  
  ORCID: [0000-0002-2948-1384](https://orcid.org/0000-0002-2948-1384)

- **Amal Saleh**  
  School of Medicine, College of Health Sciences, Addis Ababa University, Ethiopia

- **Udunna C. Anazodo**  
  Medical Artificial Intelligence Laboratory (MAI Lab), Lagos, Nigeria  
  Lawson Health Research Institute, London, Ontario, Canada  
  Montreal Neurological Institute, McGill University, Montréal, Canada  
  Department of Clinical & Radiation Oncology, University of Cape Town, South Africa  
  ORCID: [0000-0001-8864-035X](https://orcid.org/0000-0001-8864-035X)

- **Gizeaddis Lamesgin Simegn**  
  School of Biomedical Engineering, Jimma University, Ethiopia  
  ORCID: [0000-0003-1333-4555](https://orcid.org/0000-0003-1333-4555)


## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Training](#training)
5. [Prediction](#prediction)
6. [Example Usage](#example-usage)
## Prerequisites
1. 
Python >=3.10
The model architecture was implemented using Keras framework with Tensor-Flow 2.11.0, with CUDA 11.2 on an NVIDIA Tesla T4 GPU.

## Installation

2. Clone the repository:
   ```bash/cmd/terminal
   git clone https://github.com/CAMERA-MRI/SPARK2023/VIU-Net
   cd VIU-Net

   pip install -r requirements.txt```

3. Dataset Preparation
The dataset needs to be converted from 3D to 2D slices and then split into training and validation sets

```from VIU_Net import VIUNet```

# Create an instance of VIUNet
```model = VIUNet()```
# Prepare the dataset
```model.prepare(dataset_path, destination_path, split_path)```

. dataset_path: The path where the raw dataset is located.
. destination_path: The path where the prepared dataset (converted from 3D to 2D) will be saved.
. split_path: The path where the training and validation split datasets will be stored.
4. Training
Train the model using the prepared dataset.
# Train the model
```model.train(split_path, model_saving_path)```

. split_path: The path where the training and validation datasets are stored (from the previous step).
. model_saving_path: The path where the trained model and training history will be saved.
5. Prediction
Use the trained model to make predictions on new 2D images.
# Predict using the trained model
```prediction = model.predict(model_path, image)```

. model_path: The path where the trained model is saved.
. image: The 2D image on which you want to perform segmentation.

#6. Example Usage
```
#Here is a complete example of how to use the VIU-Net model:

from VIU_Net import VIUNet

VIUNet = VIUNet() #ceating object from VIUNet class to use the functions defined in it.

#Prepare the dataset
VIUNet.prepare(dataset_path="path/to/raw/dataset", 
              destination_path="path/to/processed/dataset", 
              split_path="path/to/split/dataset")


#Train the model
VIUNet.train(split_path="path/to/split/dataset", 
            model_saving_path="path/to/save/model")


#Predict on a new image
result = VIUNet.predict(model_path="path/to/save/model", 
                       image="path/to/2D/image")
```
