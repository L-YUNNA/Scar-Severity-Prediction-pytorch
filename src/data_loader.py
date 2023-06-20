import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class CombineDataset(Dataset):
    def __init__(self, frame, id_col, label_name, path_imgs, transform=None):
        """
        Args:
            frame (pd.DataFrame): Frame with the tabular data.
            id_col (string): Name of the column that connects image to tabular data
            label_name (string): Name of the column with the label to be predicted
            path_imgs (string): path to the folder where the images are.
            transform (callable, optional): Optional transform to be applied
                on a sample, you need to implement a transform to use this.
        """
        self.frame = frame
        self.id_col = id_col
        self.label_name = label_name
        self.path_imgs = path_imgs
        self.transform = transform

    def __len__(self):
        return (self.frame.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get images
        img_name = self.frame[self.id_col].iloc[idx]
        path = os.path.join(self.path_imgs, img_name)              # train_dataset = CombineDataset(train_df, 'img_name', 'Cls', train_img_dir, transform=tf)
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)

        # get features
        feats = [feat for feat in self.frame.columns if feat not in [self.label_name, self.id_col]]
        feats = np.array(self.frame[feats].iloc[idx])
        feats = torch.from_numpy(feats.astype(np.float32))

        # get label
        label = np.array(self.frame[self.label_name].iloc[idx])
        label = torch.from_numpy(label.astype(np.int64))

        return image, feats, label


class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len