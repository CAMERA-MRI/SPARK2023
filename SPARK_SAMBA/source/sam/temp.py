import os
import random
from collections import defaultdict 
import numpy as np
import json

embedding_path = "/scratch/guest190/BraTS_data_africa/"
ground_truth_path = "/scratch/guest190/BraTS_data_africa/mask/"



# temp_cases = os.listdir("/scratch/guest190/BraTS_data_africa/flair/z/embeddings")

all_cases = {}
case_idxs = defaultdict(lambda:[])
idx = 0
for d in ["z"]:
    for m in ["flair", "t1", "t2", "t1ce"]:
        temp_cases = os.listdir(f"/scratch/guest190/BraTS_data_africa/{m}/{d}/embeddings")
        for c in temp_cases:
          name = c.split("_")[0]
          case_idxs[name].append(idx)
          all_cases[idx] = embedding_path + m + "/" + d + "/embeddings/" + c
        #   print(idx)
          idx+=1


all_cases_names = case_idxs.keys()
print("all_cases_idxs", len(all_cases_names))

def load_split(all_cases, selected_cases, case_idxs):
    """return a dict with case_name as keys and slice link as value 

    Args:
        all_cases (_type_): _description_
        selected_cases (_type_): _description_
        case_idxs (_type_): _description_

    Returns:
        _type_: _description_
    """
    res = {}
    final_idx=0
    for case_name in selected_cases:
        idxs = case_idxs[case_name]
        for idx in idxs:
            res[final_idx] = all_cases[idx]
            final_idx += 1
    return res

def filter_split_black_masks(train_list):
  ground_truth_path = "/scratch/guest190/BraTS_data_africa/mask/"
  with open('/home/guest190/export_data_africa/data.json', 'r') as fp:
    data = json.load(fp)

  res = {}
  for idx in train_list:
    modality = train_list[idx].split("/")[5]
    name = train_list[idx].split("/")[7]
    name = name.split(".")[0]
    name, slice_num = name.split("_")

    # print("[name][modality][int(idx)]", [name],[modality],[int(slice_num)])

    if data[name][modality][slice_num]:
      res[idx] = train_list[idx]
  final_res = {}
  idx=0
  for x in res:
    final_res[idx]=res[x]
    idx+=1
  return final_res
  
train_cases = random.sample(all_cases_names, int(0.7*len(all_cases_names)))
train_list = load_split(all_cases, train_cases, case_idxs)#{idx:all_cases[key] for idx, key in enumerate(train_cases)}

train_cases_dict = {name:1 for name in train_cases}
val_cases = [i for i in all_cases_names if i not in train_cases_dict]
val_list = load_split(all_cases, val_cases, case_idxs)

train_list2 = filter_split_black_masks(train_list)
val_list2 = filter_split_black_masks(val_list)


def combine_modalities(lst):
  names = {}
  for idx in lst:
    name = lst[idx].split("/")[7]
    names[name] = idx
  
  res = {}
  idx = 0
  for name in names:
    res[idx] = name
    idx+=1
  return res

train_list3 = combine_modalities(train_list2)
val_list3 = combine_modalities(val_list2)

print("len(train_cases)", len(train_cases))
print("len(train_list.keys())", len(train_list.keys()))
print("len(train_list2.keys())", len(train_list2.keys()))
print("len(train_list3.keys())", len(train_list3.keys()))
print("train_list[0]", train_list3[0])

print("len(val_cases)", len(val_cases))
print("len(val_list.keys())", len(val_list.keys()))
print("len(val_list2.keys())", len(val_list2.keys()))
print("len(val_list3.keys())", len(val_list3.keys()))
# print(all_cases)

# mask_path = self.ground_truth_path + self.list_of_data[idx].split("/")[5] + "/" +  self.list_of_data[idx].split("/")[7]
