from data_preparation import prepare_dataset
from main import main
from data_preprocessing.preprocessor import Preprocessor
from utils.utils import get_task_code
from postprocessing import prepare_predictions, to_lbl

import os
from glob import glob
import torch
import time

def run_inference(data_path, parameters, output_path, ckpts_path):

    # Preparing dataset
    prepare_dataset(data_path, False) # Testing
    print("Finished prepping all gli data!")

    # Preprocessing dataset
    start = time.time()
    Preprocessor(parameters).run()
    end = time.time()
    print(f"Pre-processing time: {(end - start):.2f}")

    # Making predictions and post-processing
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for i, ckpt in enumerate(os.listdir(ckpts_path)):
        parameters["ckpt_path"] = os.path.join(ckpts_path, ckpt)
        parameters["fold"] = i
        main(parameters)

    # Post processing : Ensembling + To_label
    pred_path = output_path
    os.makedirs(pred_path, exist_ok=True)
    print(parameters["results"])
    preds = sorted(glob(f'{parameters["data"]}/predictions*'))
    examples = list(zip(*[sorted(glob(f"{p}/*.npy")) for p in preds]))
    print("Preparing final predictions")
    for e in examples:
        prepare_predictions(e, data_path, pred_path)
    print("Finished!")