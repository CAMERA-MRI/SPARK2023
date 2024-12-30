
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave
import cv2
from tensorflow.keras.layers import InputLayer,Lambda,Attention, Layer,MaxPooling2D, add,Dropout,Conv2D,Conv1D,UpSampling2D, Conv3D, BatchNormalization, Activation,MaxPool2D,MaxPool3D,Input,Conv3DTranspose,Conv2DTranspose,Concatenate
from tensorflow.keras.models import Model,Sequential
import tensorflow.keras as keras
from tensorflow.keras.applications import VGG19
import copy
import tensorflow as tf
from tensorflow.keras import backend as K
import os
from PIL import Image as im
import split_folders  ##split_folders==0.2.0
import pickle
from keras.models import load_model
## *********************** DATASET PREPARATION ************************************************************************
class VIUNet:

    def prepare(self,dataset_path, destination_path, split_path):
        t1c_list = sorted(glob.glob(dataset_path+"/*/*t1c.nii.gz"))
        t2w_list = sorted(glob.glob(dataset_path+"/*/*t2w.nii.gz"))
        t2f_list = sorted(glob.glob(dataset_path+"/*/*t2f.nii.gz"))
        seg_list = sorted(glob.glob(dataset_path+"/*/*seg.nii.gz"))
        counter = 0
        for img in range(len(t1c_list)):
            print("Processing image and mask: number", img)
            temp_img_t1c = nib.load(t1c_list[img]).get_fdata()
            # temp_img_t1c = scaler.fit_transform(temp_img_t1c.reshape(-1, temp_img_t1c.shape[-1])).reshape(temp_img_t1c.shape)

            temp_img_t2w = nib.load(t2w_list[img]).get_fdata()
            # temp_img_t2w = scaler.fit_transform(temp_img_t2w.reshape(-1, temp_img_t2w.shape[-1])).reshape(temp_img_t2w.shape)

            temp_img_t2f = nib.load(t2f_list[img]).get_fdata()
            # temp_img_t2f = scaler.fit_transform(temp_img_t2f.reshape(-1, temp_img_t2f.shape[-1])).reshape(temp_img_t2f.shape)

            temp_seg = nib.load(seg_list[img]).get_fdata()
            temp_seg = temp_seg.astype(np.uint8)

            temp_img_t1c = temp_img_t1c[60:135]
            temp_img_t2w = temp_img_t2w[60:135]
            temp_img_t2f = temp_img_t2f[60:135]
            temp_seg = temp_seg[60:135]

            for i in range(temp_seg.shape[0]):  #generating 2D image by taking one slice iteratively a long the sagittal view.
                _2D_img_t1c = temp_img_t1c[i] ## 2d images (128,128,1) depth1
                _2D_img_t2w = temp_img_t2w[i]
                _2D_img_t2f = temp_img_t2f[i]
                _2D_seg = temp_seg[i]

                # temp_combined_images = np.dstack([_2D_img_t1c, _2D_img_t2w,_2D_img_t2f]) ## 2D (128,128,3) depth3
                temp_combined_images = (_2D_img_t1c+_2D_img_t2w+_2D_img_t2f)
                ##normalization
                temp_combined_images = (temp_combined_images - np.min(temp_combined_images)) / (np.max(temp_combined_images) - np.min(temp_combined_images))

                val, counts = np.unique(temp_seg, return_counts=True) ## to calculate existence of important information
                if(1-counts[0]/(counts.sum()))>=0.05: # if there is at least 5% important information (ratio of image over blank space)
                    print("Saved")
                    _2D_seg = to_categorical(_2D_seg, num_classes = 4)
                    #saving the prepared images and mask
                    np.save(destination_path+"/images/"+str(counter)+'.npy', temp_combined_images)
                    np.save(destination_path+"/masks/"+str(counter)+'.npy', _2D_seg)
                    counter+=1
                else:
                    print("Skipped: Labeled region ratio is under 5%")
        #Splitting the image into training and validation
        input_folder = destination_path
        output_folder = split_path
        split_folders.ratio(input_folder, output = output_folder, seed=42, ratio=(.70,.30))


##**************  MODEL ARCHITECTURE *****************************************************************
    def vgg19_features(self,input):

        base_vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(None,None,3))

        x = input
        x = Activation('relu')(x)

        for l in base_vgg19.layers[9:-2]:
            x = l(x)
            x = Conv2D(512, 1, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = UpSampling2D(size=(4, 4))(x)
        return x

    def VIUNet_model(self,input_size=(128, 128, 1)):
        inputs = Input(input_size)

        # Contracting path
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)

        ## Infusing VGG19
        drop4 = self.vgg19_features(conv4)



        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # Bottom
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)


        # Expanding path

        up6 = UpSampling2D(size=(2, 2))(conv5)
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
        #**

        merge6 = Concatenate(axis=3)([drop4, up6])

        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = UpSampling2D(size=(2, 2))(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
        attention_conv3 = Attention()([conv3, conv3])
        merge7 = Concatenate(axis=3)([conv3, up7])
        # merge7 = Concatenate(axis=3)([attention_conv3, up7])

        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = UpSampling2D(size=(2, 2))(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
        attention_conv2 = Attention()([conv2, conv2])
        # merge8 = Concatenate(axis=3)([attention_conv2, up8])
        merge8 = Concatenate(axis=3)([conv2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = UpSampling2D(size=(2, 2))(conv8)
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
        attention_conv1 = Attention()([conv1, conv1])
        # merge9 = Concatenate(axis=3)([attention_conv1, up9])
        merge9 = Concatenate(axis=3)([conv1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        # Output layer
        outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=outputs)

        return model


###**** Evaluation Metrics ************************************************ 
    def dice_coef(self,y_true, y_pred):
        smooth = 1e-15
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

    def dice_loss(self,y_true, y_pred):
        return 1.0 - self.dice_coef(y_true, y_pred)

##***********DATASET LOADER FOR TRAINING ************************************
## custome data generator

    def load_img(self, img_dir, img_list,is_image):
        images = []
        for i, image_name in enumerate(img_list):
            image = np.load(img_dir+image_name)
            if is_image:
                image = image[56:184,13:141]
                # image = np.reshape(image,(128,128,1))
                images.append(image)

            else:
                image = image[:,:,2] # class 2
                image = image[56:184,13:141]
                image = np.reshape(image, (128,128,1))
                images.append(image)
        images = np.array(images)
        return images
    def imageLoader(self,img_dir, img_list, mask_dir, mask_list, batch_size):
        l = len(img_list)
        while True:
            batch_start = 0
            batch_end = batch_size
            while batch_start<l:
                limit = min(batch_end,l)
                x = self.load_img(img_dir,img_list[batch_start:limit],True)
                y = self.load_img(mask_dir, mask_list[batch_start:limit],False)

                yield(x,y)
                batch_start+=batch_size
                batch_end+=batch_size

#*************TRAINING **********************************************************
    def train(self,split_path, model_saving_path):
        #loading the model
        model = self.VIUNet_model()
        #loading the datset
        train_img_dir = split_path+"/train/images/"
        train_mask_dir =split_path+"train/masks/"
        train_img_list = os.listdir(train_img_dir)
        train_mask_list = os.listdir(train_mask_dir)

        val_img_dir = split_path+"/val/images/"
        val_mask_dir =split_path+"/val/masks/"
        val_img_list = os.listdir(val_img_dir)
        val_mask_list = os.listdir(val_mask_dir)

        batch_size =64
        
        train_img_generator = self.imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
        val_img_generator = self.imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)


        steps_per_epoch = len(train_img_list)//batch_size
        val_steps_per_epoch = len(val_img_list)//batch_size

        ##Defining early stoping mechanism and saving checkpoints
        callback  = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=0,
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_saving_path+"/VIUNet_model_ckpt.keras",
            save_weights_only=False,
            save_freq="epoch",
            monitor='val_loss',
            verbose=0,
            save_best_only=False,
            mode='auto',
            )
        #optimizer and learning rate
        lr = 0.0001
        optimizer = keras.optimizers.Adam(lr)
        #compiling the model 
        model.compile(optimizer = optimizer, loss = self.dice_loss,metrics=[self.dice_coef])
        history = model.fit(train_img_generator,
                        epochs = 200,
                        steps_per_epoch=steps_per_epoch,
                        validation_data = val_img_generator,
                        validation_steps=val_steps_per_epoch,
                        callbacks = [callback,model_checkpoint_callback])
        ##saving history and model after training
        with open(model_saving_path+'/history', 'wb') as file:
            pickle.dump(history.history, file)

        model.save(model_saving_path+'VIUNet_model.keras')

    def predict(self,model_path, image):
        ##input must have similar size as the size of the image used in training
        input_img = np.reshape(image, (1,128,128,1))
        model = load_model(model_path, custom_objects = {"dice_loss":self.dice_loss,'dice_coef':self.dice_coef})
        output = model.predict(input_img)
        cv2.imwrite(model_path/+"output.jpg",output[0])
        print("Output saved at", model_path+"output.jpg")

'''Example

# Step 1: Prepare the dataset
model.prepare(dataset_path="path/to/raw/dataset", 
              destination_path="path/to/processed/dataset", 
              split_path="path/to/split/dataset")

# Step 2: Train the model
model.train(split_path="path/to/split/dataset", 
            model_saving_path="path/to/save/model")
'''
# Step 3: Predict on a new image
VIUNet = VIUNet() #ceating object from VIUNet class to use the functions defined in it.

result =VIUNet.predict(model_path="models/viunet_EnhancingTumor.keras", 
                       image="sample/image.npy")

