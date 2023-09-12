# ---------------------------------------------------
# Import general libraries
import numpy as np
import time
from tqdm import tqdm
import os
import sys
import logging

# Import github scripts
import data_loader as dl
import utils.modelZoo_monai as mZoo
from optinet.loss import LossBraTS_UNN

# import torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# Import MONAI libraries                <--- CLEAN UP THESE IMPORTS ONCE WE KNOW WHAT libraries are used
import monai
from monai.config import print_config
from monai.data import ArrayDataset, decollate_batch, DataLoader

from monai.metrics import DiceMetric, LossMetric, HausdorffDistanceMetric
from monai.losses import DiceFocalLoss

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose
)
from monai.inferers import sliding_window_inference
from monai.utils.misc import set_determinism

# Other imports (unsure)

# ---------------------------------------------------

"""General Setup: 
    logging, utils.args, seed, cuda, root dir
"""

logger = logging.getLogger(__name__)
args = dl.get_main_args()
#---------------------------------
# import argparse
# import os 
# class Args(argparse.Namespace):
#     # data="/scratch/guest187/Data/val_SSA/monai"
#     # data = "/Users/alexandrasmith/Desktop/Workspace/Projects/UNN_BraTS23/data/val_SSA/monai/"
#     data="C:\\Users\\amoda\\Documents\\SPARK\\BraTS2023\\CC\\Backup_2407\\val_SSA\\monai\\"
#     preproc_set="val"
#     data_used="SSA"
#     # results='/Users/alexandrasmith/Desktop/Workspace/Projects/UNN_BraTS23/data/val_SSA/results/'
#     # results='/scratch/guest187/Data/val_SSA/results/monai_test/'
#     results='C:\\Users\\amoda\\Documents\\SPARK\\BraTS2023\\CC\\Backup_2407\\val_SSA\\results\\monai_test\\'
#     optimiser="adam"
#     criterion="dice"
#     exec_mode="predict"
#     seed=42
#     batch_size=4
#     val_batch_size=2
#     # ckpt_path='/scratch/guest187/Data/train_all/results/test_fullRunThrough/best_metric_model_fullTest.pth'
#     # ckpt_path='/Users/alexandrasmith/Desktop/Workspace/Projects/UNN_BraTS23/data/best_metric_model_fullTest.pth'
#     ckpt_path='C:\\Users\\amoda\\Documents\\SPARK\\BraTS2023\\CC\\Backup_2407\\Results\\train_all_monai\\test_fullRunThrough\\best_metric_model_fullTest.pth'
#     model="unet"
# args=Args()
#----------------------------
set_determinism(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = args.data
results_dir = args.results

logger.info(f"Setting up. Working from folder: {root_dir}. \nSaving to folder: {results_dir}.")
logger.info(f"\nWorking with dataset: {args.data_used}.")

# Save checkpoints
def save_checkpoint(model, last_epoch, best_acc=0, dir_add=results_dir, args=args):
    filename=f"chkpt_{args.run_name}_{args.data_used}.pt"
    state_dict = model.state_dict()
    save_dict = {"last epoch": last_epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    logger.info("\nSaving checkpoint", filename)

# ---------------------------------------------------
"""
Define model architecture:
        Done before data loader so that transforms has n_channels for EnsureShapeMultiple
"""
def define_model(checkpoint=None):
    logger = logging.getLogger(__name__)
    model_mapping = {
        'unet': mZoo.unet(),
        'dynUnet': mZoo.dynUnet(),
        'optinet': mZoo.OptiNet()# replace mZoo.another_function with the actual function
        }
    model_name = args.model
    model=model_mapping.get(model_name)
    model.to(device)
    if args.model == "unet":
        n_layers = len(model.channels)
    elif args.model == "dynUnet" or args.model == 'optinet':
        n_layers = len(model.filters)
    else:
        n_layers = 6
    logger.info(f"Number of channels: {n_layers}")

    if checkpoint != None:
        ckpt = torch.load(checkpoint, map_location=device)
        if args.model=='unet':
            model.load_state_dict(ckpt)
        if args.model=='dynUnet':
            sdict = dict(ckpt["state_dict"])
            model.load_state_dict(sdict, strict=False)
        else:
            logger.info("No checkpoint found, starting from scratch")

    return model, n_layers

# ---------------------------------------------------
# SET UP TRAINING
# ---------------------------------------------------

"""
Setup transforms, dataset
"""
def define_dataloaders(n_layers):
    data_transform = dl.define_transforms(n_layers)       # Define transforms
    dataloaders = dl.load_data(args, data_transform)        # Load data; also saves json splitData
    # train_loader, val_loader = dataloaders['train'], dataloaders['val']
    return dataloaders

"""Create Model Params:
    optimiser
    loss fn
    lr
"""
def model_params(args, model):
    logger = logging.getLogger(__name__)
    # Define optimiser
    if args.optimiser == "adam":
        optimiser = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
        logger.info("Adam optimizer set")
    elif args.optimiser == "novo":
        optimiser = monai.optimizers.Novograd(params=model.parameters(), lr=args.learning_rate)
    else:
        logger.info("Error, no optimiser provided")

    # Define loss function
    if args.criterion == 'brats':
        loss = LossBraTS_UNN
        criterion = loss(focal=True)
        logger.info("BraTS Loss set")
    elif args.criterion == "ce":
        criterion = nn.CrossEntropyLoss()
        logger.info("Cross Entropy Loss set")
    elif args.criterion == "dice":
        criterion = DiceFocalLoss(squared_pred=True, to_onehot_y=False, sigmoid=True)
        logger.info("Focal-Dice Loss set")
    else:
        logger.info("Error, no loss fn provided")

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    
    return optimiser, criterion, lr_scheduler

def compute_loss(preds, label, criterion):
    loss = criterion(preds, label)
    for i, pred in enumerate(preds[1:]):
        downsampled_label = nn.functional.interpolate(label, pred.shape[2:])
        loss += 0.5 ** (i + 1) * criterion(pred, downsampled_label)
    c_norm = 1 / (2 - 2 ** (-len(preds)))
    return c_norm * loss
# ---------------------------------------------------
# SET UP VALIDATION 
# ---------------------------------------------------

"""Key Validation functions
    metrics
    post trans ???????
    inference
"""
def val_params():
    VAL_AMP = True
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True, num_classes=4)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=True, num_classes=4)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    return VAL_AMP, dice_metric, dice_metric_batch, post_trans

# define inference method
def inference(VAL_AMP, model, input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(128,128,128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
            mode='gaussian'
        )
    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)
    
# ---------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------
"""Define training loop:
    1. initialise empty lists for val
    2. Add GradScalar which uses automatic mixed precision to accelerate training
    3. forward and backward passes
    4. validate training epoch
"""
def train(args, model, device, train_loader, val_loader, optimiser, criterion, lr_scheduler):
    logger = logging.getLogger(__name__)
    logger.info("Starting Training")
    VAL_AMP, dice_metric, dice_metric_batch, post_trans = val_params()

    # Train model --> see MONAI notebook examples
    val_interval = 1
    epoch_loss_list, val_epoch_loss_list= [], []

    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]

    metric_values = []
    metric_values_0, metric_values_1, metric_values_2, metric_values_3 = [], [], [], []

    scaler = GradScaler()

    total_start = time.time()
    model = model.to(device)
    for epoch in range(args.epochs):
        epoch_start = time.time()

        model.train()
        epoch_loss = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)
        progress_bar.set_description(f"Training Epoch {epoch}")

        for step, batch_data in progress_bar:
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            logger.info(f"\n{inputs.shape}")
            optimiser.zero_grad()

            with autocast(): # cast tensor to smaller memory footprint to avoid OOM
                outputs = model(inputs)
                logger.info(f"\n{len(outputs)}")
                # loss = compute_loss(outputs, labels, criterion)
                loss = criterion.forward(outputs, labels)

            # Calculate Loss and Update optimiser using scalar
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            epoch_loss += loss.item()
            
            progress_bar.set_postfix({
                "bat_train_loss" : loss.item(), 
                "Ave_train_loss" : epoch_loss/(step + 1)
            })

            epoch_loss2 = epoch_loss/(step+1)
            lr_scheduler.step()
        epoch_loss_list.append(epoch_loss2)
        logger.info(f"\nEpoch {epoch} average loss: {epoch_loss2:.3f}")
        
        #Run validation for current epoch
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            progress_bar = tqdm(enumerate(val_loader), total=len(val_loader),dynamic_ncols=True)
            progress_bar.set_description(f"Val_train Epoch {epoch}")

            for step, batch in enumerate(val_loader):
                val_inputs, val_labels = batch[0].to(device), batch[1].to(device)
                
                with torch.no_grad():
                    val_outputs = inference(VAL_AMP, model, val_inputs)
                    # val_loss = compute_loss(val_outputs, val_labels, criterion)
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
            last_epoch = [metric, epoch_loss2, val_epoch_loss, val_epoch_loss / (step + 1) ]
            metric_batch = dice_metric_batch.aggregate()
            logger.info(f"{metric}")
            logger.info("{metric_batch}")
            metric_0 = metric_batch[0][0].item()
            metric_1 = metric_batch[0][1].item()
            metric_2 = metric_batch[0][2].item()
            metric_3 = metric_batch[0][3].item()
            metric_values_0.append(metric_0)
            metric_values_1.append(metric_1)
            metric_values_2.append(metric_2)
            metric_values_3.append(metric_3)

            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                logger.info(f"\nNew best metric model")

        logger.info(
            f"\ncurrent epoch: {epoch + 1} current mean dice: {metric:.3f}"
            f"\nMean Dice per Region is: label 1: {metric_1:.4f};  label 2: {metric_2:.3f} label 3: {metric_3:.3f}"
            f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
        )
        save_checkpoint(
                    model,
                    last_epoch,
                    best_acc=best_metric,
                )
        logger.info(f"Total time for epoch {epoch + 1} is: {(time.time() - epoch_start):.3f}")

    total_time = time.time() - total_start
    logger.info(f"Training completed. Total time taken: {total_time}")