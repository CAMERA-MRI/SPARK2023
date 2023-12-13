import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import nibabel as nib
import matplotlib.image
import random
from ultralytics import YOLO
import matplotlib.pyplot as plt


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Subset

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
from skimage import transform, io, segmentation


from monai.networks.nets import UNet

import monai

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_optimizers():
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4)
    return [optimizer], [scheduler]


class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class

    def forward(self, input_):
      return self.net(input_)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, verbose=True)
        # return [optimizer], [scheduler]
        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"}

    def prepare_batch(self, batch):
        return (batch["input"], batch["mask"])



    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss


def get_yolo_box(image, yolo):
  results = yolo.predict(image, stream = False)
  boxes = results[0].boxes

  if len(boxes.xyxy.cpu().numpy()) == 0:
    return [0,0,0,0]

  box = boxes.xyxy[0].cpu().numpy()
  box = [int(i) for i in box]

  return box



def get_3d_from_SAM(_3d_image, sam_model, decoder ,device, flair, yolo):
  sam_model.to(device)

  output = np.zeros_like(_3d_image)
  sclices = _3d_image.shape[2]
  for s in range(sclices):
    image = _3d_image[:,:,s]


    image = np.stack((image, image, image), axis = -1 )
    image = np.uint8(image) #sure?

    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    image = sam_transform.apply_image(image)
    image = torch.as_tensor(image.transpose(2, 0, 1)).to('cuda')
    image = sam_model.preprocess(image[None,:,:,:]) # (1, 3, 1024, 1024)
    image = image[0,...] # (3, 1024, 1024)


    t = flair[:,:,s]
    t = np.stack((t, t, t), axis = -1 )
    t = np.uint8(t)
    bbox = get_yolo_box(t, yolo)

    if bbox == [0,0,0,0]:
      masks = np.zeros((240,240))
    else:
      box_np = np.array(bbox)
      sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
      box = sam_trans.apply_boxes(box_np, (256, 256))
      box_torch = torch.as_tensor(box, dtype=torch.float)

      with torch.no_grad():
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=None, boxes=box_torch.to(device), masks=None)

      with torch.no_grad():
        emb = sam_model.image_encoder(image[None,...].to(device))[0]

      with torch.no_grad():
        mask_predictions, _ = decoder(
              image_embeddings= emb[None,...].to(device), # (B, 256, 64, 64)
              image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
              sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
              dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
              multimask_output=False,
            )

      masks = sam_model.postprocess_masks(mask_predictions, (1024, 1024), (240, 240))
      masks = masks > sam_model.mask_threshold
      masks = np.squeeze(masks.cpu().numpy(), axis = 0)
      masks = np.int32(masks)
      masks = np.squeeze(masks, axis = 0)

    output[:,:,s] = masks

  return output


def get_final_masks(yolo_path, sam_path, sam_decoder_path, voting_path, data_path, save_path):
  #---------------------------------------Models------------------------------------------#

  yolo = YOLO(yolo_path)

  model_type = 'vit_h'
  checkpoint = sam_path
  sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)

  decoder = sam_model.mask_decoder
  model_path = sam_decoder_path
  checkpoint = torch.load(model_path, map_location="cpu")
  decoder.load_state_dict(checkpoint)
  decoder.to(device)

  net = UNet(
    spatial_dims=3,
    in_channels=8,
    out_channels=3,
    channels=(64, 128, 256),
    strides=(2, 2),
  ).to(device)
  loss_function = monai.losses.DiceLoss(sigmoid=True, to_onehot_y = False).to(device)
  optimizer = torch.optim.Adam(net.parameters(), 0.001)
  model = Model.load_from_checkpoint(voting_path,
                                   net=net,
                                   criterion=loss_function,
                                   learning_rate=0.001,
                                   optimizer_class=torch.optim.Adam
                                   )

  model = model.to(device)

  data = os.listdir(data_path)
  for i in tqdm(data):
    t2f =  nib.load(data_path + i + "/" + i +"-t2f.nii.gz")
    t2f = t2f.get_fdata()
    lower_bound, upper_bound = np.percentile(t2f, 0.5), np.percentile(t2f, 99.5)
    image_data_pre = np.clip(t2f, lower_bound, upper_bound)
    d = (np.max(image_data_pre)-np.min(image_data_pre))
    d = 1 if d==0 else d
    image_data_pre = ((image_data_pre - np.min(image_data_pre))/d)
    image_data_pre[t2f==0] = 0
    t2f = image_data_pre


    t1n =  nib.load(data_path + i + "/" + i + "-t1n.nii.gz")
    t1n = t1n.get_fdata()
    lower_bound, upper_bound = np.percentile(t1n, 0.5), np.percentile(t1n, 99.5)
    image_data_pre = np.clip(t1n, lower_bound, upper_bound)
    d = (np.max(image_data_pre)-np.min(image_data_pre))
    d = 1 if d==0 else d
    image_data_pre = ((image_data_pre - np.min(image_data_pre))/d)
    image_data_pre[t1n==0] = 0
    t1n = image_data_pre


    t2w =  nib.load(data_path + i + "/" + i + "-t2w.nii.gz")
    t2w = t2w.get_fdata()
    lower_bound, upper_bound = np.percentile(t2w, 0.5), np.percentile(t2w, 99.5)
    image_data_pre = np.clip(t2w, lower_bound, upper_bound)
    d = (np.max(image_data_pre)-np.min(image_data_pre))
    d = 1 if d==0 else d
    image_data_pre = ((image_data_pre - np.min(image_data_pre))/d)
    image_data_pre[t2w==0] = 0
    t2w = image_data_pre


    t1c =  nib.load(data_path + i + "/" + i + "-t1c.nii.gz")
    t1c = t1c.get_fdata()
    lower_bound, upper_bound = np.percentile(t1c, 0.5), np.percentile(t1c, 99.5)
    image_data_pre = np.clip(t1c, lower_bound, upper_bound)
    d = (np.max(image_data_pre)-np.min(image_data_pre))
    d = 1 if d==0 else d
    image_data_pre = ((image_data_pre - np.min(image_data_pre))/d)
    image_data_pre[t1c==0] = 0
    t1c = image_data_pre


    sam_flair = get_3d_from_SAM(t2f, sam_model, decoder, device, t2f*255., yolo)
    sam_t1 = get_3d_from_SAM(t1n, sam_model, decoder,device, t2f*255., yolo)
    sam_t2 = get_3d_from_SAM(t2w, sam_model, decoder,device, t2f*255., yolo)
    sam_t1c = get_3d_from_SAM(t1c, sam_model, decoder,device, t2f*255., yolo)



    input_ = np.stack(( t2f, t1n, t2w, t1c, sam_t1, sam_t2, sam_flair, sam_t1c), axis = 0)
    input_ = np.float32(input_) 
    input_ = np.expand_dims(input_, axis = 0)
    input_ = torch.as_tensor(input_, dtype=torch.float ).to(device)


    with torch.no_grad():
      out_mask = model.net(input_)

    m = out_mask
    m =torch.sigmoid(m).cpu().numpy()

   
    m = np.squeeze(m, axis = 0)
    m = m[:,:,:,:155]

    output = np.zeros(m.shape[1:])

    msk1 = m[0]
    msk1[msk1<0.5]= 0
    msk1[msk1>=0.5]= 1

    msk2 = m[1]
    msk2[msk2>=0.5]= 1
    msk2[msk2<0.5]= 0

    msk3 = m[2]
    msk3[msk3>=0.5]= 1
    msk3[msk3<0.5]= 0


    output[msk2>0] = 2
    output[msk3>0] = 3
    output[msk1>0] = 1

    id = i.split("-")[2]
    time = i.split("-")[3]
    ni_img = nib.Nifti1Image(output, affine=np.eye(4))
    nib.save(ni_img, save_path + "/seg-" + id + "-" + time + ".nii.gz" )

