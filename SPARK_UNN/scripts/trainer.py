from monai_functions import *
from utils.utils import get_main_args, set_cuda_devices, set_granularity
from pytorch_lightning.strategies import DDPStrategy

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
"""
General Setup: 
    logging, utils.args, seed, cuda, root dir
"""

current_datetime = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
log_file_name = f"trainin_{current_datetime}.log"
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_name)

args = get_main_args()

set_determinism(args.seed)
set_granularity()

set_cuda_devices(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#----------------------- SETUP -----------------------

model, n_channels = define_model(args.ckpt_path)                            #get model and print out model architecture
with open(f'{args.run_name}-{args.exec_mode}_modelArch.txt', 'w') as f:
    print(model, file=f)

dataloaders = define_dataloaders(n_channels)                                #load datasets
train_loader, val_loader = dataloaders['train'], dataloaders['val']
optimiser, criterion, lr_scheduler = model_params(args, model)              #set model params

#----------------------- TRAIN MODEL -----------------------
train(args, model, device, train_loader, val_loader, optimiser, criterion, lr_scheduler)