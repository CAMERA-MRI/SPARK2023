import torch
import torch.nn as nn
import torch.optim as optim
from nnunet.nn_unet import NNUnet
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, RichProgressBar
from utils.utils import *
from data_loader import load_data

def main():
    args = get_main_args()
    # set_granularity()
    set_cuda_devices(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    #     seed_everything(args.seed)
    # data_module = DataModule(args)
    # data_module.setup()
    ckpt_path = verify_ckpt_path(args)

    if ckpt_path is not None:
        model = NNUnet.load_from_checkpoint(ckpt_path, strict=False, args=args)
    else:
        model = NNUnet(args)
        
    callbacks = [RichProgressBar(), ModelSummary(max_depth=2)]

    # what does this do?
    # if args.exec_mode == "train" and args.save_ckpt:
    #     callbacks.append(
    #         ModelCheckpoint(
    #             dirpath=f"{args.ckpt_store_dir}/checkpoints",
    #             filename="{epoch}-{dice:.2f}",
    #             monitor="dice",
    #             mode="max",
    #             save_last=True,
    #         )
    #     )

    dataloaders = load_data(args.data, args.batch_size)

    # Define loss function and optimizer
    if args.criterion == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    if args.optimiser == 'adam':
        # fill in params
        optimser = optim.Adam(model.parameters())
    elif args.optimiser == 'sgd':
        optimiser = optim.SGD(model.parameters(), lr=args.learning_rate)

    # is this part necessary? set_cuda_devices?
    model = model.to(device)

    # TRAIN your model
    # run this on train and val data splits
    if args.exec_mode == "train":
        for epoch in range(args.num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                # Keep track for print statements and logging
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimiser.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        # if isinstance(outputs, tuple):
                        #     outputs = outputs[0] # extract tensor if output is a tuple
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimiser.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'Epoch {epoch + 1}/{args.num_epochs}, {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # EVALUATE your model
    if args.exec_mode == "evaluate":
        model.eval()
        # Images as well as ground truth labels are avaiable
        for inputs, labels in dataloaders['eval']:
            # Get predictions
            outputs = model(inputs)
            # Perform evaluation calculations

    # Make predictions with your model
    if args.exec_mode == "predict":
        model.eval()
        for batch in dataloaders['pred']:
            # pred dataloader does not contain any labels
            inputs = batch
            outputs = model(inputs)
            # Process the predictions


if __name__ == "__main__":
    main()


"""
Typical training loop for a NN from Pytorch classifier [tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html):
"""

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0): # loop over the samples in the dataset 
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


