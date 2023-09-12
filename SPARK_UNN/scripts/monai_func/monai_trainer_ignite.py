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

"""Define model architecture:
        Done before data loader so that transforms has n_channels for EnsureShapeMultiple
"""
model=UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    # channels=(32, 64, 128, 256, 320, 320), #nnunet channels, deoth 6
    # channels=(64, 96, 128, 192, 256, 384, 512) # optinet, depth 7
    strides=(2, 2, 2, 2), # length should = len(channels) - 1
    # kernel_size=,
    # num_res_units=,
    # dropout=0.0,
    ).to(device)
n_channels = len(model.channels)
print(n_channels)

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

"""Create supervised_trainer using ignite:
        Ignite is a high-level library to help with training and evaluating neural networks in PyTorch flexibly and transparently. It provides a higher level of abstraction over PyTorch and helps to maximize the reproducibility of your experiments. Ignite helps you write compact but full-featured training loops in a few lines of code, while you remain in full control of the entire process. It also provides a simple way to define and handle training and validation metrics.
"""
scaler = GradScaler()
with autocast(enabled=True):
    trainer = ignite.engine.create_supervised_trainer(model, optimiser, criterion, device, False)

""" 
LOGGING TRAINING
Setup event handlers for checkpointing and logging
    adding checkpoint handler to save models (network params and optimizer stats) during training
    StatsHandler prints loss at every iteration
        user can also customize print functions and can use output_transform to convert engine.state.output if it's not a loss value
    TensorBoardStatsHandler plots loss at every iteration
"""
log_dir = os.path.join(args.results, "logs")
checkpoint_handler = ignite.handlers.ModelCheckpoint(log_dir, "net", n_saved=10, require_empty=False)
trainer.add_event_handler(
    event_name=ignite.engine.Events.EPOCH_COMPLETED,
    handler=checkpoint_handler,
    to_save={"net": model, "opt": optimiser},
)

train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
train_stats_handler.attach(trainer)

train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: x)
train_tensorboard_stats_handler.attach(trainer)

"""Validation every N epochs for model validation during training"""
validation_every_n_epochs = 1

# Set parameters for validation
metric_name = "Mean_Dice"

# add evaluation metric to the evaluator engine
val_metrics = {metric_name: MeanDice()}
post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
post_label = Compose([AsDiscrete(threshold=0.5)])

"""Ignite evaluator expects batch=(img, seg) 
    and returns output=(y_pred, y) at every iteration,
    user can add output_transform to return other values"""
evaluator = ignite.engine.create_supervised_evaluator(
    model,
    val_metrics,
    device,
    True,
    output_transform=lambda x, y, y_pred: (
        [post_pred(i) for i in decollate_batch(y_pred)],
        [post_label(i) for i in decollate_batch(y)],
    ),
)

@trainer.on(ignite.engine.Events.EPOCH_COMPLETED(every=validation_every_n_epochs))
def run_validation(engine):
    evaluator.run(val_loader)

# Add stats event handler to print validation stats via evaluator
val_stats_handler = StatsHandler(
    name="evaluator",
    # no need to print loss value, so disable per iteration output
    output_transform=lambda x: None,
    # fetch global epoch number from trainer
    global_epoch_transform=lambda x: trainer.state.epoch,
)
val_stats_handler.attach(evaluator)

# add handler to record metrics to TensorBoard at every validation epoch
val_tensorboard_stats_handler = TensorBoardStatsHandler(
    log_dir=log_dir,
    # no need to plot loss value, so disable per iteration output
    output_transform=lambda x: None,
    # fetch global epoch number from trainer
    global_epoch_transform=lambda x: trainer.state.epoch,
)
val_tensorboard_stats_handler.attach(evaluator)

"""add handler to draw the first image and the corresponding label and model output in the last batch
    here we draw the 3D output as GIF format along Depth axis, at every validation epoch"""
val_tensorboard_image_handler = TensorBoardImageHandler(
    log_dir=log_dir,
    batch_transform=lambda batch: (batch[0], batch[1]),
    output_transform=lambda output: output[0],
    global_iter_transform=lambda x: trainer.state.epoch,
)
evaluator.add_event_handler(
    event_name=ignite.engine.Events.EPOCH_COMPLETED,
    handler=val_tensorboard_image_handler,
)

"""Run training loop"""
total_start = time.time()
# use amp to accelerate training
max_epochs = args.epochs
state = trainer.run(train_loader, max_epochs)
total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")


# Save checkpoint
# TODO