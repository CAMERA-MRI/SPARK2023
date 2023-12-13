import numpy as np
import os
from tqdm import tqdm


def test(loc):
    files = os.listdir(loc)
    for file in tqdm(files):
        try:
            f = np.load(os.path.join(loc, file))
            f.shape
        except Exception as exc:
            print(f"case {file} faild, why? {exc}")

base = "/scratch/guest190/BraTS_data_africa"
views = ["x", "y", "z"]
domains = ["flair", "t1", "t1ce", "t2"]

for view in views:
    for domain in domains:
        print(f"{domain}/{view}/embeddings")
        loc = f"{base}/{domain}/{view}/embeddings"
        test(loc)

        print("___"*5)