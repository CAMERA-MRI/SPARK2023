import os
from glob import glob
from subprocess import call

import nibabel as nib
import numpy as np
from scipy.ndimage.measurements import label


def to_lbl(pred):
    enh = pred[2]

    pad = pred == 0.5
    pred[pad==True] = 0

    c1, c2, c3 = pred[0] > 0.4, pred[1] > 0.35, pred[2] > 0.375

    pred = (c1 > 0).astype(np.uint8)
    pred[(c2 == False) * (c1 == True)] = 2 
    pred[(c3 == True) * (c1 == True)] = 3

    components, n = label(pred == 3)
    for et_idx in range(1, n + 1):
        _, counts = np.unique(pred[components == et_idx], return_counts=True)
        if 1 < counts[0] and counts[0] < 4 and np.mean(enh[components == et_idx]) < 0.9:
            pred[components == et_idx] = 1

    et = pred == 3
    if 0 < et.sum() and et.sum() < 5 and np.mean(enh[et]) < 0.9:
        pred[et] = 1

    pred = np.transpose(pred, (2, 1, 0)).astype(np.uint8)
    return pred


def prepare_predictions(e, data_path, output_path):
    fname = e[0].split("/")[-1].split(".")[0]
    preds = [np.load(f) for f in e]
    p = to_lbl(np.mean(preds, 0))

    img = nib.load(os.path.join(data_path,fname, f"{fname}.nii.gz"))
    nib.save(
        nib.Nifti1Image(p, img.affine, header=img.header),
        os.path.join(os.path.join(output_path, fname + ".nii.gz")),
    )
