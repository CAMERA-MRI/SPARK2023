# saved 2d data
print("2d multinode multiclass")
import os, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Subset

from tqdm import tqdm
# from torchsummary import summary
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import wandb
import nibabel as nib
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
from skimage import transform, io, segmentation
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import random
from collections import defaultdict 

torch.manual_seed(2023)
np.random.seed(2023)



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




model_type = 'vit_h'
checkpoint = 'sam_vit_h_4b8939.pth'
# sam_model = sam_model_registry[model_type](checkpoint=checkpoint)#.to('cuda')



embedding_path = "/scratch/guest190/BraTS_data_africa/"
ground_truth_path = "/scratch/guest190/BraTS_data_africa/mask/"

all_cases = {}
case_idxs = defaultdict(lambda:[])
idx = 0
for d in ["x", "y", "z"]:
    for m in ["flair", "t1", "t2", "t1ce"]:
        temp_cases = os.listdir(f"/scratch/guest190/BraTS_data_africa/{m}/{d}/embeddings")
        for c in temp_cases:
          name = c.split("_")[0]
          case_idxs[name].append(idx)
          all_cases[idx] = embedding_path + m + "/" + d + "/embeddings/" + c
        #   print(idx)
          idx+=1


# all_cases_names = case_idxs.keys()
all_cases_names = list(case_idxs.keys())
print("all_cases_idxs", len(all_cases_names))

def load_split(all_cases, selected_cases, case_idxs):
    res = {}
    final_idx=0
    for case_name in selected_cases:
        idxs = case_idxs[case_name]
        for idx in idxs:
            res[final_idx] = all_cases[idx]
            final_idx += 1
    return res

# train_cases = random.sample(all_cases_names, int(0.7*len(all_cases_names)))
train_cases = all_cases_names[:int(0.7*len(all_cases_names))]
train_list = load_split(all_cases, train_cases, case_idxs)#{idx:all_cases[key] for idx, key in enumerate(train_cases)}

train_cases_dict = {name:1 for name in train_cases}
val_cases = [i for i in all_cases_names if i not in train_cases_dict]
val_list = load_split(all_cases, val_cases, case_idxs)

print("len(train_cases)", len(train_cases))
print("len(train_list.keys())", len(train_list.keys()))

print("len(val_cases)", len(val_cases))
print("len(val_list.keys())", len(val_list.keys()))

class BraTSDataset(Dataset):

    def __init__(self, ground_truth_path, embedding_path, list_of_data, sam_model, device='cuda'):
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
        self.sam_model = sam_model

    def __len__(self):
        return len(self.list_of_data)

    def __getitem__(self, idx):


      embedding = np.load(self.list_of_data[idx])
      embedding = torch.as_tensor(embedding, dtype=torch.float)#.cuda(self.device)


      #--------------------------------------------------

      mask = np.load(self.ground_truth_path + self.list_of_data[idx].split("/")[5] + "/" +  self.list_of_data[idx].split("/")[7])
    #   mask[mask>0] = 1
      size = mask.shape
      box = get_bounding_box(mask)
      # mask = squarify(mask)



      mask = transform.resize(
                      mask,
                      (256, 256),
                      order=0,
                      preserve_range=True,
                      mode="constant",
                  )
      class_1 = np.zeros_like(mask)
      class_2 = np.zeros_like(mask)
      class_3 = np.zeros_like(mask)

      class_1[np.where(mask == 1)] = 1
      class_2[np.where(mask == 2)] = 1
      class_3[np.where(mask == 3)] = 1

      mask = mask = np.stack((class_1, class_2, class_3), axis = 0 )

    

      mask = torch.as_tensor(mask)#.cuda(self.device)
    #   mask = mask[None,...]


      # box = get_bounding_box(mask)
      # box = [0,0,mask.shape[-2],mask.shape[-1]]
      box_np = np.array(box)
      sam_trans = ResizeLongestSide(self.sam_model.image_encoder.img_size)
      box = sam_trans.apply_boxes(box_np, (size[0], size[1]))
      box_torch = torch.as_tensor(box, dtype=torch.float)#.cuda(self.device)
      # box_torch = box_torch[None, :]

      with torch.no_grad():
      # embedding = sam_model.image_encoder(image)
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
          points=None,
          boxes=box_torch,
          masks=None,
      )


      sample = {'embedding': embedding,
                'mask': mask,
                # 'image_pe' : sam_model.prompt_encoder.get_dense_pe(),
                'sparse_embeddings': torch.squeeze(sparse_embeddings, axis = 0),
                'dense_embeddings': torch.squeeze(dense_embeddings, axis = 0),
                  }

      return sample

def eval(val_dataloader, sam_model, seg_loss, gpu, image_pe):
  loss = 0
  for step_val, data in enumerate(val_dataloader):
    embedding, mask, sparse_embeddings, dense_embeddings = data["embedding"], data["mask"], data["sparse_embeddings"], data["dense_embeddings"]


    with torch.no_grad():
      # predicted masks
      mask_predictions, _ = sam_model(
      image_embeddings= embedding.cuda(gpu), # (B, 256, 64, 64)
      image_pe=image_pe.cuda(gpu), # (1, 256, 64, 64)
      sparse_prompt_embeddings=sparse_embeddings.cuda(gpu), # (B, 2, 256)
      dense_prompt_embeddings=dense_embeddings.cuda(gpu), # (B, 256, 64, 64)
      multimask_output=True,
      )

      loss += seg_loss(mask_predictions, mask.cuda(gpu))
  loss /= step_val+1
  return loss

def train(args):

    hprams = {
      "num_epochs": 15,
      "best_loss": 1e10,
      "model_save_path": "/scratch/guest190/models/SAM_multi",
      "lr":1e-5,
      "weight_decay":0,

    }

    ngpus_per_node = torch.cuda.device_count()

    # Get the rank of the current process
    SLURM_NODEID = int(os.environ.get("SLURM_NODEID"))

    # Get the local ID of the current process
    SLURM_LOCALID = int(os.environ.get("SLURM_LOCALID"))

  #------------------------------------
    rank = SLURM_NODEID * ngpus_per_node + SLURM_LOCALID
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.init_method,
        world_size=args.world_size,
        rank=rank
    )

    print("From Rank: {} Initializing Process Group...".format(rank))

  #-------------------------------------
    gpu = SLURM_LOCALID
    torch.cuda.set_device(gpu)


    model_type = 'vit_h'
    checkpoint = 'sam_vit_h_4b8939.pth'
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)#.cuda(gpu)
    get_dense_pe = sam_model.prompt_encoder.get_dense_pe()

    train = BraTSDataset(ground_truth_path, embedding_path, train_list,sam_model , gpu)
    val = BraTSDataset(ground_truth_path, embedding_path, val_list, sam_model, gpu)

    train_sampler = DistributedSampler(train, num_replicas=args.world_size, rank=rank, shuffle=False, drop_last=False)
    val_sampler = DistributedSampler(val, num_replicas=args.world_size, rank=rank, shuffle=False, drop_last=False)


    train_loader = DataLoader(train, batch_size= 8, shuffle=False, num_workers=0, sampler=train_sampler)
    val_loader = DataLoader(val, batch_size= 4, shuffle=False, num_workers=0, sampler=val_sampler)

    model_save_path = hprams['model_save_path']
    num_epochs = hprams['num_epochs']
    best_loss = hprams['best_loss']
    lr = hprams['lr']
    weight_decay = hprams['weight_decay']


    os.makedirs(model_save_path, exist_ok=True)

    model = sam_model.mask_decoder
    model.cuda(gpu)


    model.train()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=0.25,  patience=5)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean').cuda(gpu)

    wandb.init(project = "SAM Finetune", 
           save_code = True, group= 'Decoder_multiclass_1')

    for epoch in range(num_epochs):
        epoch_loss = 0
        # train
        for step, data in enumerate(tqdm(train_loader)):
            embedding, mask, sparse_embeddings, dense_embeddings = data["embedding"], data["mask"], data["sparse_embeddings"], data["dense_embeddings"]
            # print("embedding", embedding.shape)
            # print("mask", mask.shape)
            # # print("image_pe", image_pe.shape)
            # print("sparse_embeddings", sparse_embeddings.shape)
            # print("dense_embeddings", dense_embeddings.shape)
            # with torch.no_grad():

            #   sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            #       points=None,
            #       boxes=box,
            #       masks=None,
            #   )

            # predicted masks
            mask_predictions, _ = model(
              image_embeddings= embedding.cuda(gpu), # (B, 256, 64, 64)
              image_pe=get_dense_pe.cuda(gpu), # (1, 256, 64, 64)
              sparse_prompt_embeddings=sparse_embeddings.cuda(gpu), # (B, 2, 256)
              dense_prompt_embeddings=dense_embeddings.cuda(gpu), # (B, 256, 64, 64)
              multimask_output=True,
            )

            loss = seg_loss(mask_predictions, mask.cuda(gpu))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # break
        epoch_loss /= step+1
        wandb.log({"loss": epoch_loss})
        scheduler.step(epoch_loss)
        val_loss = None
        if val_loader is not None: val_loss = eval(val_loader, model, seg_loss, gpu, get_dense_pe)
        print(f'EPOCH: {epoch}, Loss: {epoch_loss}, Val loss {val_loss}')
        wandb.log({"val_loss": val_loss})
        # wandb.log({"epoch": epoch, "loss": loss})

        # save the latest model checkpoint
        # torch.save(sam_model.state_dict(), os.path.join(model_save_path, 'sam_model_'+str(epoch)+'.pth'))

        # save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.module.state_dict(), os.path.join(model_save_path, 'sam_model_best.pth'))



def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-n', '--nodes', default=1,
    #                     type=int, metavar='N')
    # parser.add_argument('-g', '--gpus', default=1, type=int,
    #                     help='number of gpus per node')
    # parser.add_argument('-nr', '--nr', default=0, type=int,
    #                     help='ranking within the nodes')

    parser.add_argument('-max_epochs', type=int, default=30, help="Maximum number of epochs")
    parser.add_argument('-num_workers', type=int, default=0, help="Number of workers")
    parser.add_argument('-init_method', default='tcp://127.0.0.1:3456', type=str, help="Initialization method")
    parser.add_argument('-dist_backend', default='nccl', type=str, help="Distributed backend")
    parser.add_argument('-world_size', default=1, type=int, help="World size")
    parser.add_argument('-distributed', action='store_true', help="Enable distributed training")




    args = parser.parse_args()

    # args.world_size = args.gpus * args.nodes
    # os.environ['MASTER_ADDR'] = ''
    # os.environ['MASTER_PORT'] = '8888'
    # mp.spawn(train, args=(args,))
    train(args)


if __name__ == "__main__":
  main()