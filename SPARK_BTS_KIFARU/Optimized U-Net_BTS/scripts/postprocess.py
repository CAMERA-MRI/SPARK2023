import os
from glob import glob
from subprocess import call
import nibabel as nib
import numpy as np
from scipy.ndimage.measurements import label

def to_lbl(pred):
    enh = pred[2]
    c1, c2, c3 = pred[0] > 0.4, pred[1] > 0.35, pred[2] > 0.375
    pred = (c1 > 0).astype(np.uint8)
    pred[(c2 == False) * (c1 == True)] = 2
    pred[(c3 == True) * (c1 == True)] = 4

    components, n = label(pred == 4)
    for et_idx in range(1, n + 1):
        _, counts = np.unique(pred[components == et_idx], return_counts=True)
        if 1 < counts[0] and counts[0] < 4 and np.mean(enh[components == et_idx]) < 0.9:
            pred[components == et_idx] = 1

    et = pred == 4
    if 0 < et.sum() and et.sum() < 5 and np.mean(enh[et]) < 0.9:
        pred[et] = 1

    pred = np.transpose(pred, (2, 1, 0)).astype(np.uint8)
    return pred

def prepare_preditions(e):
    fname = e[0].split("/")[-1].split(".")[0]
    preds = [np.load(f) for f in e]
    p = to_lbl(np.mean(preds, 0))

    img_path = f"/scratch/guest189/hackathon_data/BraTS2021_val/images/{fname}.nii.gz"
    if os.path.exists(img_path):
        img = nib.load(img_path)
        nib.save(nib.Nifti1Image(p, img.affine, header=img.header), os.path.join("/scratch/guest189/hackathon_data/results/final_preds281", fname + ".nii.gz"))
    else:
        print(f"File not found: {img_path}")

os.makedirs("/scratch/guest189/hackathon_data/results/final_preds281")
preds = sorted(glob(f"/scratch/guest189/hackathon_data/results/'predictions_epoch=28-dice=89_25_task=01_fold=0_tta'*"))
examples = list(zip(*[sorted(glob(f"{p}/*.npy")) for p in preds]))
# print("Preparing final predictions")
for e in examples:
    prepare_preditions(e)
print("Finished!")