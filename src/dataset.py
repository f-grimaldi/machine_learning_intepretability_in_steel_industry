import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import os
import cv2
from PIL import Image

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

def train_test_split(metadata_root, val_size):
    df = pd.read_csv(metadata_root)
    val_index = np.random.choice(np.arange(0, df.shape[0]), int(val_size*df.shape[0]), replace=False)
    df_train = df.iloc[df.index.isin(val_index) == False, :]
    df_val = df.iloc[val_index, :]
    return df_train, df_val


class SteelDataset(object):
    def __init__(self, metadata_root, image_root,
                 shape=(256, 1600, 3), colors=None,
                 transform=None, mask_transform = None):

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
        path = '{}/{}'.format(self.image_root, self.metadata.ImageId[idx])
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
