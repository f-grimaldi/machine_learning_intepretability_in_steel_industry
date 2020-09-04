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
from albumentations import (
    VerticalFlip, HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)

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
        path = '{}/{}'.format(self.image_root,
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

"""
Get all the batches using the SteelDataset and put them into one single huge matrix
"""
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


"""
Take the matrix generated using SteelMatrix and divide each image in 6 parts
"""
class SlidingWindow():

    def __init__(self, X, y, M):
        self.X, self.y, self.M = X, y, M

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):

        X, y, M = torch.Tensor([]), [], torch.Tensor([]).long()

        for i in range(6):
            #print(self.X[idx, :, i*64:(i+1)*64].shape)
            X = torch.cat([X, self.X[idx, :, :, i*64:(i+1)*64].view(1, 3, 64, 64)])
            M = torch.cat([M, self.M[idx, :, i*64:(i+1)*64].view(1, 64, 64)])
            if self.M[idx, :, i*64:(i+1)*64].sum() == 0:
                y.append(0)
            else:
                y.append(self.y[idx])

        return X, torch.tensor(y), M

"""
Augment the under represent class (target_classes = [1, 2, 4]) n_times
(times_to_augment = [1, 1, 1]) by using HorizontalFlip and VerticalFlip
with probability p (p = 1/sqrt(2)).
"""
class Augmentator(object):
    def __init__(self, target_classes = [1,2,4],
                 times_to_augment = [1, 1, 1],
                 p = 1/(2**0.5), augmentations=None):
        self.target_classes = target_classes
        self.times_to_augment = times_to_augment
        self.compose = self.get_augmentations(augmentations, p)

    def get_augmentations(self, augmentations, p):
        if augmentations:
            return augmentations
        else:
            return Compose([HorizontalFlip(p=p), VerticalFlip(p=p)])

    def __call__(self, X, y, M):
        aug_X, aug_M, aug_y = torch.Tensor([]), torch.tensor([]).long(), []
        for c, t in zip(self.target_classes, self.times_to_augment):
            tmp_X = X[y==c]
            tmp_M = M[y==c]
            for i in range(t):
                for sample, sample_mask in tqdm(zip(tmp_X, tmp_M)):
                    image, mask = np.transpose(sample.numpy(), (1, 2, 0)), sample_mask.numpy()
                    data = {"image": image, 'mask': mask}
                    augmented = self.compose(**data)
                    image_tensor = torch.tensor(np.transpose(augmented['image'], (2, 0, 1)))
                    mask_tensor = torch.tensor(augmented['mask'])
                    aug_X = torch.cat([aug_X, image_tensor.view(1, *list(image_tensor.shape))])
                    aug_M = torch.cat([aug_M, mask_tensor.view(1, *list(mask_tensor.shape))])
                    aug_y.append(c)
        return aug_X, torch.tensor(aug_y).long(), aug_M
