# 2d input 3d mask



print("Bounding box, flair_z")
import os
import torch
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Subset

from tqdm import tqdm
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
# import wandb
import nibabel as nib
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
from skimage import transform, io, segmentation
import random


torch.manual_seed(2023)
np.random.seed(2023)

# change here
data_path = "/content/out"


def squarify(M,val = 0):
    (a,b)=M.shape
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    return np.pad(M,padding,mode='constant',constant_values=val)



def get_bounding_box(mask):

  y_indices, x_indices = np.where(mask > 0)
  if len(y_indices) == 0:
    return [0,0,0,0]
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = mask.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  # bboxes = np.array([x_min, y_min, x_max, y_max])

  return [x_min, y_min, x_max, y_max]


def get_points(mask):
  mask= mask.cpu().numpy()
  mask = np.squeeze(mask, axis = 0)
  x = np.where(mask>0)
  if len(x[0]) == 0:
    return (0,0)
  all_points = list(zip(x[0], x[1]))
  return random.choices(all_points)


def dice_loss(y_true, y_pred):
    """
    implementation for generalized dice score loss
    GDL = 1 - (2 * sum(intersection)/(sum(pred)+sum(true)))
    """

    # y_pred = tf.round(y_pred)
    intersection = torch.sum(y_true * y_pred)
    pred_sum = torch.sum( y_pred)
    true_sum = torch.sum( y_true)
    smooth = torch.ones_like(intersection)*1e-5
    return 1-((2*intersection+smooth)/(pred_sum+true_sum+smooth))


model_type = 'vit_h'
checkpoint = 'sam_vit_h_4b8939.pth'
sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to('cuda')



embedding_path = "/scratch/guest190/BraTS_data/flair/z/embeddings/"
ground_truth_path = "/scratch/guest190/kaggle"

train_list = os.listdir(embedding_path)
train_list = train_list[:int(len(train_list)*0.7)]
val_list = train_list[int(len(train_list)*0.7):]

class BraTSDataset(Dataset):

    def __init__(self, ground_truth_path, embedding_path, list_of_data, device='cuda'):
        """
        Args:
            embedding_path (string): Path to  images.
            masks_path (string): path to masks.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ground_truth_path = ground_truth_path
        self.device = device
        self.list_of_data = list_of_data
        self.embedding_path = embedding_path

    def __len__(self):
        return len(self.list_of_data)

    def __getitem__(self, idx):


      embedding = np.load(self.embedding_path + "/" + self.list_of_data[idx])
      embedding = torch.as_tensor(embedding, dtype=torch.float, device=self.device)


      #--------------------------------------------------
      case_id = self.list_of_data[idx].split("_")[1]
      slice_ = int(self.list_of_data[idx].split("_")[-1].split(".")[0])
      # "BraTS2021_00000"
      case_name = "BraTS2021_"+ case_id
      img =  nib.load(self.ground_truth_path + "/" + case_name + "/"+ case_name +"_seg.nii.gz")
      img = img.get_fdata()
      mask = img[:,:,slice_]

      mask[mask>0] = 1
      # mask = squarify(mask)



      mask = transform.resize(
                      mask,
                      (256, 256),
                      order=0,
                      preserve_range=True,
                      mode="constant",
                  )

        # class_1 = np.zeros_like(mask)
        # class_2 = np.zeros_like(mask)
        # class_3 = np.zeros_like(mask)

        # class_1[np.where(mask == 1)] = 1
        # class_2[np.where(mask == 2)] = 1
        # class_3[np.where(mask == 4)] = 1

        # mask = mask = np.stack((class_1, class_2, class_3), axis = -1 )


      box = get_bounding_box(mask)
      # box = [0,0,mask.shape[-2],mask.shape[-1]]
      box_np = np.array(box)
      sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
      box = sam_trans.apply_boxes(box_np, (mask.shape[-2], mask.shape[-1]))
      box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)

      mask = torch.as_tensor(mask).to(self.device)
      mask = mask[None,...]


      sample = {"embedding": embedding,
                  'mask': mask,
                  'box' : box_torch
                  }

      return sample


train = BraTSDataset(ground_truth_path, embedding_path, train_list)
val = BraTSDataset(ground_truth_path, embedding_path, val_list)



train_loader = DataLoader(train, batch_size= 2, shuffle=True, num_workers=0)
val_loader = DataLoader(val, batch_size= 1, shuffle=True, num_workers=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



hprams = {
    "num_epochs": 10,
    "best_loss": 1e10,
    "model_save_path": "/scratch/guest190/models",
    "lr":1e-4,
    "weight_decay":0,
    "device": device,
    "train_dataloader": train_loader,
    "val_dataloader": val_loader
    }

def eval(val_dataloader, sam_model, seg_loss):
  loss = 0
  for step_val, data in enumerate(val_dataloader):
    embedding, mask, box = data["embedding"], data["mask"], data["box"]

    with torch.no_grad():
      # embedding = sam_model.image_encoder(image)
      sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
          points=None,
          boxes=box,
          masks=None,
      )

      # predicted masks
      mask_predictions, _ = sam_model.mask_decoder(
      image_embeddings= embedding.to(device), # (B, 256, 64, 64)
      image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
      sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
      dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
      multimask_output=False,
      )

      loss += seg_loss(mask_predictions, mask.to(device))
  loss /= step_val+1
  return loss



def train(**kwargs):
    model_save_path = kwargs['model_save_path']
    num_epochs = kwargs['num_epochs']
    best_loss = kwargs['best_loss']
    lr = kwargs['lr']
    weight_decay = kwargs['weight_decay']
    device = kwargs['device']
    train_dataloader = kwargs['train_dataloader']
    # embedding_path = kwargs['embedding_path']
    val_dataloader = kwargs['val_dataloader']

    os.makedirs(model_save_path, exist_ok=True)

    sam_model.train()
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=0.25,  patience=3)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')


    for epoch in range(num_epochs):
        epoch_loss = 0
        # train
        for step, data in enumerate(tqdm(train_dataloader)):
            mask, box, embedding =  data["mask"], data["box"], data["embedding"]

            embedding.to(device)

            with torch.no_grad():

              sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                  points=None,
                  boxes=box,
                  masks=None,
              )

            # predicted masks
            mask_predictions, _ = sam_model.mask_decoder(
              image_embeddings= embedding.to(device), # (B, 256, 64, 64)
              image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
              sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
              dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
              multimask_output=False,
            )

            loss = seg_loss(mask_predictions, mask.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # break
        epoch_loss /= step+1
        scheduler.step(epoch_loss)
        val_loss = None
        if val_dataloader is not None: val_loss = eval(val_dataloader, sam_model, seg_loss)
        print(f'EPOCH: {epoch}, Loss: {epoch_loss}, Val loss {val_loss}')
        # wandb.log({"epoch": epoch, "loss": loss})

        # save the latest model checkpoint
        # torch.save(sam_model.state_dict(), os.path.join(model_save_path, 'sam_model_'+str(epoch)+'.pth'))

        # save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(sam_model.state_dict(), os.path.join(model_save_path, 'sam_model_best.pth'))




train(**hprams)