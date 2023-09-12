import json
import os
import pickle
from subprocess import run
from joblib import Parallel, delayed
import ctypes
import numpy as np
import torch

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

def set_cuda_devices(args):
    assert args.gpus <= torch.cuda.device_count(), f"Requested {args.gpus} gpus, available {torch.cuda.device_count()}."
    device_list = ",".join([str(i) for i in range(args.gpus)])
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", device_list)

def set_granularity():
    cuda_path = os.getenv('CUDA_PATH')
    libcudart_path = os.path.join(cuda_path, 'lib64', 'libcudart.so')
    _libcudart = ctypes.CDLL(libcudart_path)
    # _libcudart = ctypes.CDLL("libcudart.so")
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128
    
def run_parallel(func, args):
    return Parallel(n_jobs=-1)(delayed(func)(arg) for arg in args)

def extract_imagedata(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)

def positive_int(value):
    ivalue = int(value)
    assert ivalue > 0, f"Argparse error. Expected positive integer but got {value}"
    return ivalue

def non_negative_int(value):
    ivalue = int(value)
    assert ivalue >= 0, f"Argparse error. Expected non-negative integer but got {value}"
    return ivalue

# added function
def float_0_1(value):
    fvalue = float(value)
    assert 0 <= fvalue <= 1, f"Argparse error. Expected float value to be in range (0, 1), but got {value}"
    return fvalue

def get_config_file(args):
    if args.data != "/data":
        path = os.path.join(args.data, "config.pkl")
    else:
        task_code = args.datasets
        path = os.path.join(args.data, task_code, "config.pkl")
    return pickle.load(open(path, "rb"))

def get_main_args(strings=None):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    
    # Data set allocatin and execution param
    arg("--preproc_set", type=str,
        default="training", choices=["training", "val", "test"],
        help="Mode for data preprocessing"
    )
    arg("--exec_mode", type=str,
        default="train", choices=["train", "evaluate", "predict"],
        help="Execution mode to run the model"
    )
    arg("--task", type=str, 
        default="other", choices=["data_prep", "other"], 
        help="Mode for dataset class call"
    )

    arg("--run_name", type=str, help="Run name")

    #For file loading (paths & static vars)
    # Folders
    arg("--data", type=str, default="/data", help="Path to main data directory")
    arg("--procData", type=str, default="/data", help="Path for saving output directory")
    arg("--results", type=str, default="/results", help="Path to results directory")
    arg("--ckpt_path", type=str, default=None, help="Path for loading checkpoint")
    arg("--ckpt_store_dir", type=str, default="/results", help="Path for saving checkpoint")
    arg("--resume_training", action="store_true", help="Resume training from the last checkpoint")

    # Naming conventions & saving
    arg("--modal", type=list, default=["t1c", "t1n", "t2w", "t2f"], help="List of modality abbreviations")
    arg("--data_used", type=str, default="all", choices=["ALL", "GLI", "SSA"], help="The set or subset of data that is used for training")
    arg("--data_grp", type=str, default="ATr", help="Dataset used",
        choices={"ATr": "BraTS23_train",
                  "AV": "BraTS23_val",
                 "ATe": "BraTS23_test",
                 "fSSATr": "FakeSSA_train",
                 "fSSAV": "FakeSSA_val",
                 "fSSATe": "FakeSSA_test",
                 "STr": "SSA_train",
                  "SV": "SSA_val",
                 "STe": "SSA_test"
        })

    arg("--config", type=str, default=None, help="Config file with arguments")          # <--- Do we need a configs file for training?
    arg("--save_preds", action="store_true", help="Enable prediction saving")
    arg("--save_ckpt", action="store_true", help="Enable saving checkpoint")

    # For preprocessing stacked nifty files
    arg("--target_shape", type=bool, default=False, help="Target shape for cropOrPad")
    arg("--ohe", action="store_true", help="Add one-hot-encoding for foreground voxels (voxels > 0)")

    arg("--augs", type=str, default=None, help="Fake SSA data transforms to apply: use resample or augment")

    # # Cluster allocations
    arg("--n_jobs", type=int, default=-1, help="Number of parallel jobs for data preprocessing")                        # <---------- CHANGE default
    arg("--gpus", type=non_negative_int, default=1, help="Number of gpus")
    arg("--nodes", type=non_negative_int, default=1, help="Number of nodes")
    arg("--num_workers", type=non_negative_int, default=8, help="Number of subprocesses to use for data loading")

    # # Training parameters ************** TO COMPLETE NEXT WEEK***************
    arg("--model", type=str, help="Model selection name; see model zoo")

    arg("--nfolds", type=positive_int, default=5, help="Number of cross-validation folds")
    arg("--fold", type=non_negative_int, default=0, help="Fold number")
    arg("--seed", type=non_negative_int, default=None, help="Random seed")
    
    arg("--epochs", type=non_negative_int, default=150, help="Number of training epochs.")
    arg("--learning_rate", type=float, default=0.0003, help="Learning rate")
    arg("--nvol", type=positive_int, default=4, help="Number of volumes which come into single batch size") # <---------- CHANGE default
    arg("--batch_size", type=positive_int, default=2, help="Batch size") # <---------- CHANGE default
    arg("--optimiser", type=str, default="adam", choices=["adam", "novo"], help="Optimiser")
    arg("--criterion", type=str, default="dice", choices=["ce", "dice", "brats"], help="Loss")

    arg("--val_batch_size", type=positive_int, default=2, help="Validation batch size")                                     # <---------- CHANGE default

    # # Other training params ************ TO CHECK IF NEEDED ******************
    arg("--gradient_clip_val", type=float, default=0, help="Gradient clipping norm value")
    arg("--negative_slope", type=float, default=0.01, help="Negative slope for LeakyReLU")
    arg("--tta", action="store_true", help="Enable test time augmentation")
    arg("--deep_supervision", action="store_true", help="Enable deep supervision")
    arg("--invert_resampled_y", action="store_true", help="Resize predictions to match label size before resampling")
    arg("--amp", action="store_true", help="Enable automatic mixed precision")
    arg("--skip_first_n_eval", type=non_negative_int, default=0, help="Skip the evaluation for the first n epochs.")
    arg("--momentum", type=float, default=0.99, help="Momentum factor")
    arg("--weight_decay", type=float, default=0.0001, help="Weight decay (L2 penalty)")
    arg("--warmup", type=non_negative_int, default=5, help="Warmup iterations before collecting statistics")
    arg("--min_fmap", type=non_negative_int, default=4, help="Minimal dimension of feature map in the bottleneck")
    arg("--deep_supr_num", type=non_negative_int, default=2, help="Number of deep supervision heads")
    # added
    arg("--layout", type=str, default="NCDHW")

    arg(
        "--norm",
        type=str,
        choices=["instance", "instance_nvfuser", "batch", "group"],
        default="instance",
        help="Normalization layer",
    )
    arg(
        "--oversampling",
        type=float_0_1,
        default=0.4,
        help="Probability of crop to have some region with positive label",
    )
    arg(
        "--overlap",
        type=float_0_1,
        default=0.25,
        help="Amount of overlap between scans during sliding window inference",
    )
    arg(
        "--blend",
        type=str,
        choices=["gaussian", "constant"],
        default="constant",
        help="How to blend output of overlapping windows",
    )

    if strings is not None:
        arg(
            "strings",
            metavar="STRING",
            nargs="*",
            help="String for searching",
        )
        args = parser.parse_args(strings.split())
    else:
        args = parser.parse_args()
        if args.config is not None:
            config = json.load(open(args.config, "r"))
            args = vars(args)
            args.update(config)
            args = Namespace(**args)


    # with open(f"{args.results}/params.json", "w") as f:
    #     json.dump(vars(args), f)

    return args

def verify_ckpt_path(args):
    if args.resume_training:
        resume_path_ckpt = os.path.join(
            args.ckpt_path if args.ckpt_path is not None else "", "checkpoints", "last.ckpt"
        )
        resume_path_results = os.path.join(args.results, "checkpoints", "last.ckpt")
        if os.path.exists(resume_path_ckpt):
            return resume_path_ckpt
        if os.path.exists(resume_path_results):
            return resume_path_results
        print("[Warning] Checkpoint not found. Starting training from scratch.")
        return None
    if args.ckpt_path is None or not os.path.isfile(args.ckpt_path):
        print(f"Provided checkpoint {args.ckpt_path} is not a file. Starting training from scratch.")
        return None
    return args.ckpt_path

## ***** ADDED INTO ONE FUNCTION CALL --> CHECK WITH ALEX AND DELETE COMMENTED LINES *****
# # Read in the dataset folder structure
# def load_dir(directory):
#     data_dir = directory
#     subjID = sorted(os.listdir(data_dir))
#     print("You are working in :", data_dir, "Total subjects: ", len(subjID))
#     subj_dir = os.path.join(data_dir, subjID)
#     data = {
#         "subjID": subjID
#     }
#     with open("data_overview.json", "w") as file:
#         json.dump(data, file)
#     return subj_dir 