from torch.utils.data import DataLoader
import dataset_original
import dataset_preprocessed

data_preprocessed_path = '/scratch/guest189/hackathon_data/BraTS2021_train/11_3d/'
data_original_path = '/scratch/guest189/hackathon_data/data/'

dataset_pre = dataset_preprocessed.data_set(data_preprocessed_path)
dataloader_pre = DataLoader(
   dataset_pre, batch_size=1, shuffle=True, num_workers=4,)

dataset_org = dataset_original.data_set(data_original_path)
dataloader_org = DataLoader(
   dataset_org, batch_size=1, shuffle=True, num_workers=4,)


print("------- preprocessed data ------------")


for i, (imgs, labels) in enumerate(dataloader_pre):
   print("imgs shape", imgs.shape, "label shape", labels.shape)
   if i > 5:
       break


print("------- original data ------------")


for i, (imgs, labels) in enumerate(dataloader_org):
   print("imgs shape", imgs.shape, "label shape", labels.shape)
   if i > 5:
       break