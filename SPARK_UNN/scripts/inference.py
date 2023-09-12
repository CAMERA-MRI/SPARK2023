from monai_functions import *
from utils.utils import get_main_args
from data_transforms import define_transforms
import torch.utils.data as data_utils
from data_class import MRIDataset
from monai.handlers.utils import from_engine
import matplotlib.pyplot as plt

args = get_main_args()
set_determinism(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# List of validation files
# Should always be the same for Synapse validation right?
validation_dir= '/scratch/guest187/Data/val_SSA'
validation_files = [os.path.join(validation_dir, file) for file in os.listdir(validation_dir)]

# checkpoint to test: /scratch/guest187/Data/train_all/results/test_run/best_metric_model.pth 
checkpoint = '/scratch/guest187/Data/train_all/results/test_fullRunThrough/best_metric_model_fullTest.pth'

# Define model architecture
model, n_channels = define_model(args.ckpt_path)

# Load validation data to dataloader
data_transforms = define_transforms(n_channels)
validation_dataset = MRIDataset(args, validation_files, transform=data_transforms['val'])
# Do you need dataloader here?
# validation_dataloader = data_utils.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

# Validation parameters
VAL_AMP, dice_metric, dice_metric_batch, post_transforms = val_params()

# Perform inference
model.eval()

# "Evaluate on original image spacing"
with torch.no_grad():
    for val_data in validation_dataloader:
        val_inputs = val_data["image"].to(device)
        val_data["pred"] = inference(val_inputs)
        val_data = [post_transforms(i) for i in decollate_batch(val_data)]
        val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
        dice_metric(y_pred=val_outputs, y=val_labels)
        dice_metric_batch(y_pred=val_outputs, y=val_labels)

    metric_org = dice_metric.aggregate().item()
    metric_batch_org = dice_metric_batch.aggregate()

    dice_metric.reset()
    dice_metric_batch.reset()

metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()

print("Metric on original image spacing: ", metric_org)
print(f"metric_tc: {metric_tc:.4f}")
print(f"metric_wt: {metric_wt:.4f}")
print(f"metric_et: {metric_et:.4f}")

# ----------------------------------------------------------
# "Evaluate images"
with torch.no_grad():
    # select one image to evaluate and visualize the model output
    # this shouldn't work because of setup of Dataset??
    val_input = validation_dataset[6]["image"].unsqueeze(0).to(device)
    roi_size = (128, 128, 64)
    sw_batch_size = 4
    val_output = inference(val_input)
    val_output = post_transforms(val_output[0]) # val_output should be segmentation volume
# -----------------------------------------------------------

# Produce segmentations for multiple images to save and submit
with torch.no_grad():
    # Run through each image
    for img in validation_dataset:
        print(img.shape)
        val_inputs = img[0].to(device)
        # print(val_input.shape)
        val_output = inference(VAL_AMP, model, val_inputs)
        val_output = post_transforms(val_output[0])


    # Visualise (check) segmentation results
    plt.figure("output", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"output channel {i}")
        plt.imshow(val_output[i, :, :, 70].detach().cpu())
    plt.show()
    
    # Save segmentation
    # Write to results directory
    # dir = args.results