from torch.utils.data.dataset import Dataset
from torchvision import transforms
from skimage import io
import numpy as np
import pandas as pd
import torch

class MyDataset(Dataset):
    def __init__(self, csv_path, image_ids, image_folder, label_folder, nb_dates, patch_size):
        # Read the csv file
        self.data_info = pd.read_csv(str(csv_path))

        self.patch_size = patch_size
        self.nb_dates = nb_dates

        self.all_imgs = []
        for nd in self.nb_dates:
            imgs_i = []
            for city in image_ids:
                image_file = image_folder / city / f'{city}_{nd}.npy'
                imgs_i.append(np.load(image_file))
            self.all_imgs.append(imgs_i)

        self.all_labels = []
        for city in image_ids:
            label_file = label_folder / city / 'cm' / f'{city}-cm.tif'
            label = io.imread(label_file)
            label[label == 1] = 0
            label[label == 2] = 1
            self.all_labels.append(label)

        # Calculate len
        self.data_len = self.data_info.shape[0] - 1

    def __getitem__(self, index):
        x = int(self.data_info.iloc[:, 0][index])
        y = int(self.data_info.iloc[:, 1][index])
        image_id = int(self.data_info.iloc[:, 2][index])
        transformation_id = int(self.data_info.iloc[:, 3][index])

        def transform_date(patch, tr_id):
            if tr_id == 0:
               patch = patch
            elif tr_id == 1:
               patch = np.rot90(patch, k=1)
            elif tr_id == 2:
               patch = np.rot90(patch, k=2)
            elif tr_id == 3:
               patch = np.rot90(patch, k=3)

            return patch

        image_patch = []
        for nd in self.nb_dates:
            find_patch = self.all_imgs[self.nb_dates.index(nd)][image_id] [x:x + self.patch_size, y:y + self.patch_size, :]
            find_patch = np.concatenate( (find_patch[:,:,1:4], np.reshape(find_patch[:,:,7], (find_patch.shape[0],find_patch.shape[1],1))), 2) #take the 4 highest resolution channels
            image_patch.append(np.transpose(transform_date(find_patch, transformation_id), (2,0,1)))
        find_labels = self.all_labels[image_id] [x:x + self.patch_size, y:y + self.patch_size]
        label_patch = transform_date(find_labels, transformation_id)


        return np.ascontiguousarray(image_patch), np.ascontiguousarray(label_patch)

    def __len__(self):
        return self.data_len
