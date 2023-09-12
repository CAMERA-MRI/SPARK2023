# Import general libraries
import numpy as np
import time
from tqdm import tqdm
import os
import sys
import logging

# Import github scripts
import data_loader as dl

# import torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# Import MONAI libraries                <--- CLEAN UP THESE IMPORTS ONCE WE KNOW WHAT libraries are used
import monai
from monai.config import print_config
from monai.data import ArrayDataset, decollate_batch, DataLoader
from monai.handlers import (
    CheckpointLoader,
    IgniteMetric,
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
)
from monai.metrics import DiceMetric, LossMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss, DiceFocalLoss
from monai.networks import nets as monNets
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose
)
from monai.inferers import sliding_window_inference
from monai.utils import first
from monai.utils.misc import set_determinism

# Other imports (unsure)
import ignite
import nibabel

"""General Setup: 
    logging,
    utils.args 
    seed,
    cuda, 
    root dir"""
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
args = dl.get_main_args()
set_determinism(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = args.data
results_dir = args.results
"""
Potentially useful functions for model tracking and checkpoint loading
"""

def save_checkpoint(model, epoch, best_acc=0, dir_add=results_dir, args=args):
    filename=f"chkpt_{args.run_name}_{epoch}_{best_acc}.pt"
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("\nSaving checkpoint", filename)


"""Define model architecture:
        Done before data loader so that transforms has n_channels for EnsureShapeMultiple
"""
model=UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    # channels=(16, 32, 64, 128, 256),
    # channels=(32, 64, 128, 256, 320, 320), #nnunet channels, deoth 6
    channels=(64, 96, 128, 192, 256, 384, 512) # optinet, depth 7
    strides=(2, 2, 2, 2, 2, 2), # length should = len(channels) - 1
    # kernel_size=,
    # num_res_units=,
    # dropout=0.0,
    ).to(device)
n_channels = len(model.channels)
print(f"Number of channels: {n_channels}")

"""Setup transforms, dataset"""
# Define transforms
data_transform = dl.define_transforms(n_channels)
# Load data
dataloaders = dl.load_data(args, data_transform)                            # this also saves a json splitData
train_loader, val_loader = dataloaders['train'], dataloaders['val']

"""Create Model Params:
    optimiser
    loss fn
    lr
"""
# Print out model architecture
print(model)

# Load model checkpoint
# 

# Define optimiser
if args.optimiser == "adam":
    optimiser = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    print("Adam optimizer set")
elif args.optimiser == "sgd":
    optimiser = torch.optim.SGD(params=model.parameters())
    print("SGD optimizer set")
elif args.optimiser == "novo":
    optimiser = monai.optimizers.Novograd(params=model.parameters(), lr=args.learning_rate)
else:
    print("Error, no optimiser provided")

# Define loss function
if args.criterion == "ce":
    criterion = nn.CrossEntropyLoss()
    print("Cross Entropy Loss set")
elif args.criterion == "dice":
    criterion = DiceFocalLoss(squared_pred=True, to_onehot_y=False, sigmoid=True)
    print("Focal-Dice Loss set")
else:
    print("Error, no loss fn provided")

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

"""
Setup validation stuff
    metrics
    post trans ???????
    define inference
"""
VAL_AMP = True
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True, num_classes=4)
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=True, num_classes=4)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# define inference method
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 155),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            mode='gaussian'
        )
    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

"""
Define training loop
    initialise empty lists for val
    Add GradScalar which uses automatic mixed precision to accelerate training
    forward and backward passes
    validate training epoch

"""
# Train model --> see MONAI notebook examples
val_interval = 1
epoch_loss_list = []
val_epoch_loss_list = []

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]

metric_values = []
metric_values_0 = []
metric_values_1 = []
metric_values_2 = []
metric_values_3 = []

scaler = GradScaler()

total_start = time.time()

for epoch in range(args.epochs):
    epoch_start = time.time()
    # print("-" * 10)
    # print(f"epoch {epoch + 1}/{args.epochs}")
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)
    progress_bar.set_description(f"Training Epoch {epoch}")

    # for step, batch in progress_bar:
    for step, batch_data in progress_bar:
        step_start = time.time()
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimiser.zero_grad()
       
        with autocast(): # cast tensor to smaller memory footprint to avoid OOM
            """ FOR USE WITH A DIFFUSION MODEL ONLY
            # Generate random noise
            noise = torch.randn_like(images).to(device)
            Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()
            # Get model prediction
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
            loss = F.mse_loss(noise_pred.float(), noise.float())
             """

            # print(inputs.shape)
            outputs = model(inputs)
            loss = criterion.forward(outputs, labels)
        
        # Calculate Loss and Update optimiser using scalar
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        epoch_loss += loss.item()
        progress_bar.set_postfix({"bat_train_loss" : loss.item(), "Ave_train_loss" : epoch_loss/(step + 1)})
        
        print(
            f"\n{step}/{len(train_loader.dataset)//train_loader.batch_size}"
            f",     Batch train_loss: {loss.item():.4f}"
            f",     Step time: {(time.time() - step_start):.4f}"
        )
        epoch_loss2 = epoch_loss/(step+1)
        lr_scheduler.step()
    epoch_loss_list.append(epoch_loss2)
    print(f"\nEpoch {epoch} average loss: {epoch_loss2:.4f}")
    
    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_epoch_loss = 0
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader),dynamic_ncols=True)
        progress_bar.set_description(f"Val_train Epoch {epoch}")

        for step, batch in enumerate(val_loader):
            val_inputs, val_labels = batch[0].to(device), batch[1].to(device)
            """ FOR USE WITH A DIFFUSION MODEL ONLY
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            # Get model prediction
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
            val_loss = F.mse_loss(noise_pred.float(), noise.float())
            """
            with torch.no_grad():
                val_outputs = inference(val_inputs)
                val_loss = criterion.forward(val_outputs, val_labels)
                
                val_labels_list = decollate_batch(val_labels)
                val_outputs_convert = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs_convert, y=val_labels_list)
                dice_metric_batch(y_pred=val_outputs_convert, y=val_labels_list)

            val_epoch_loss += val_loss.item()
            progress_bar.set_postfix({"Val_loss": val_epoch_loss / (step + 1)})
        val_epoch_loss_list.append(val_epoch_loss / (step + 1))
        
        metric = dice_metric.aggregate()[0].item()
        metric_values.append(metric)
        metric_batch = dice_metric_batch.aggregate()
        # print(metric)
        # print(metric_batch)

        metric_0 = metric_batch[0][0].item()
        metric_values_0.append(metric_0)

        metric_1 = metric_batch[0][1].item()
        metric_values_1.append(metric_1)

        metric_2 = metric_batch[0][2].item()
        metric_values_2.append(metric_2)

        metric_3 = metric_batch[0][3].item()
        metric_values_3.append(metric_3)

        dice_metric.reset()
        dice_metric_batch.reset()

        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            best_metrics_epochs_and_time[0].append(best_metric)
            best_metrics_epochs_and_time[1].append(best_metric_epoch)
            best_metrics_epochs_and_time[2].append(time.time() - total_start)
            save_checkpoint(
                    model,
                    epoch,
                    best_acc=best_metric,
                )
            torch.save(
                model.state_dict(),
                os.path.join(results_dir, f"best_metric_model_{args.run_name}.pth"),
            )
            print("\nsaved new best metric model")
        print(
            f"\ncurrent epoch: {epoch + 1} current mean dice: {metric:.4f}"
            f"\nMean Dice per Region is: label 1: {metric_1:.4f};  label 2: {metric_2:.4f} label 3: {metric_3:.4f}"
            f"\nbest mean dice: {best_metric:.4f}"
            f" at epoch: {best_metric_epoch}"
        )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start


# Save checkpoint
# TODO