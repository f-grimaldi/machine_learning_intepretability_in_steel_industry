import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import os
import cv2

from PIL import Image
from torch.utils.data import dataloader
from torchvision import transforms
from tqdm import tqdm

class Transpose(object):
    """
    From (H, W, C) to (C, H, W)
    """
    def __call__(self, img):
        #print(np.transpose(img, (2, 1, 0)).shape)
        return np.transpose(img, (2, 1, 0))


class ToFloat(object):
    """
    Divide and cast
    """
    def __call__(self, img):
        # print(type((img//255).float()))
        # print((img//255).float().shape)

        return (img//255).float()

class Resize(object):
    """
    Resize img
    """
    def __init__(self, size=(64, 400)):
        self.size = size

    def __call__(self, img):
        return cv2.resize(img, self.size)

class SteelDataset(object):
    def __init__(self,
                 metadata_root,
                 image_root,
                 shape=(256, 1600, 3),
                 colors=None,
                 transform=None,
                 mask_transform = None):

        self.metadata_root = metadata_root
        self.image_root = image_root
        self.shape = shape
        self.metadata = pd.read_csv(self.metadata_root).sample(frac=1).reset_index(drop=True)
        self.transform = transform
        self.mask_transform = mask_transform
        self.init_colors(colors)

    def init_colors(self, colors):
        if colors:
            self.colors = colors
        else:
            steelblue = np.array((70, 130, 180))
            red = np.array((255, 0, 0))
            green = np.array((10, 155, 10))
            orange = np.array((255, 120, 10))
            self.colors = [steelblue, orange, green, red]

    def get_image(self, idx):
        path = '{}/{}/{}'.format(self.image_root,
                                 self.metadata.Folder[idx],
                                 self.metadata.ImageId[idx])
        # img = cv2.imread(path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(path)
        return img, np.array(img)

    def get_class(self, idx):
        return self.metadata.loc[idx, 'ClassId']

    def get_mask(self, idx):
        mask= np.zeros(self.shape[0]*self.shape[1]).astype(np.uint8)

        if type(self.metadata.EncodedPixels[idx]) == float:
            return mask

        array = np.asarray([int(x) for x in self.metadata.EncodedPixels[idx].split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            mask[int(start):int(start+lengths[index])] = 1
            current_position += lengths[index]

        return np.flipud(np.rot90(mask.reshape(self.shape[1], self.shape[0]),k=1))

    def get_masked_img(self, idx):
        img, data = self.get_image(idx)
        mask = self.get_mask(idx)
        label = self.get_class(idx)
        data[mask==1, 0] = self.colors[label-1][0]
        data[mask==1, 1] = self.colors[label-1][1]
        data[mask==1, 2] = self.colors[label-1][2]
        return data

    def __getitem__(self, idx):
        ### 1. Extract image/input
        img, data = self.get_image(idx)
        ### 2. Extract desired output
        target = {}
        # 2.a) Extract label
        target['label'] = self.get_class(idx)
        # 2.b) Extract segmentation
        target['mask'] = self.get_mask(idx)
        #print(target['mask'].max())
        ### 3. Transform
        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            #print('Here')
            target['mask'] = self.mask_transform(target['mask'])

        return img, target

    def __len__(self):
        return self.metadata.shape[0]

class SteelMatrix():
    def __init__(self, SteelDataset, batch_size, n_workers=4):
        self.metadata = SteelDataset.metadata
        self.shape = SteelDataset.shape
        self.data = dataloader.DataLoader(SteelDataset, batch_size, num_workers=n_workers)

    def get_matrix(self, tqdm_disable=False, return_mask = False):
        X, M, y = torch.Tensor(), torch.Tensor().long(), torch.Tensor().long()

        for batch in tqdm(self.data):
            imgs, masks, labels = batch[0], batch[1]['mask'].long(), batch[1]['label']
            if return_mask:
                X, M, y = torch.cat([X, imgs]), torch.cat([M, masks]).long(), torch.cat([y, labels]).long()
            else:
                X, y = torch.cat([X, imgs]), torch.cat([y, labels]).long()

        if return_mask:
            return X, M, y
        return X, y

    def get_masks(self, tqdm_disable=False):
        M = torch.Tensor().long()

        for batch in tqdm(self.data):
            masks  = batch[1]['mask'].long()
            M  = torch.cat([M, masks]).long()
        return M

if __name__ == '__main__':
    dataset = SteelDataset(metadata_root='../data/train.csv', image_root='../data/train_images')
    X, y = dataset.__getitem__(0)
    fig, ax = plt.subplots(3, 1, figsize=(20, 12))
    ax[0].set_title('Original image: {}'.format(dataset.metadata.iloc[0, 0]))
    ax[0].imshow(X)
    ax[1].set_title('Masked image')
    ax[1].imshow(dataset.get_masked_img(0))
    ax[2].set_title('Mask')
    ax[2].imshow(y['mask'])
    plt.show()
