import sys
import os
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')
import dataset
from dataset import SteelDataset, SteelMatrix, Transpose, Resize
from utils import get_default_params, train_val_test_split

import numpy as np

import torch
import time
import argparse

from torch.utils.data import dataloader
from torchvision import transforms


def get_argparse():
    # 1. Set argument
    parser = argparse.ArgumentParser(description='Explain SteelNetClassifier')
    # 1.a) General args
    parser.add_argument('--image_path',
                        type=str,            default='../data/train_images')
    parser.add_argument('--train_metadata_path',
                        type = str,          default='../data/binaryData/train_data.csv')
    parser.add_argument('--val_metadata_path',
                        type = str,          default='../data/binaryData/val_data.csv')
    parser.add_argument('--test_metadata_path',
                        type = str,          default='../data/binaryData/test_data.csv')
    parser.add_argument('--train_input_path',
                        type=str,            default='../data/binaryData/X_train_binary.pth')
    parser.add_argument('--train_label_path',
                        type=str,            default='../data/binaryData/y_train_binary.pth')
    parser.add_argument('--val_input_path',
                        type=str,            default='../data/binaryData/X_val_binary.pth')
    parser.add_argument('--val_label_path',
                        type=str,            default='../data/binaryData/y_val_binary.pth')
    parser.add_argument('--test_input_path',
                        type=str,            default='../data/binaryData/X_test_binary.pth')
    parser.add_argument('--test_label_path',
                        type=str,            default='../data/binaryData/y_test_binary.pth')
    parser.add_argument('--train_mask_path',
                        type=str,            default='../data/binaryData/M_train_binary.pth')
    parser.add_argument('--val_mask_path',
                        type=str,            default='../data/binaryData/M_val_binary.pth')
    parser.add_argument('--test_mask_path',
                        type=str,            default='../data/binaryData/M_test_binary.pth')


    args = parser.parse_args()
    return args


def get_transform(size, mean, std):
    img_transform = transforms.Compose([transforms.Resize(size = size), transforms.ToTensor(),
                                      transforms.Normalize(mean = mean, std = std)])
    mask_transform = transforms.Compose([dataset.Resize(size = (size[1], size[0]))])
    return img_transform, mask_transform

def create_data(images_path, metadata_path,
                batch_size=1000, num_workers= 0,
                return_mask=True, params=None):
    if not params:
        params = get_default_params()
        print('Loading default params:\n{}'.format(params))
    img_t, mask_t = get_transform(params['size'], params['mean'], params['std'])
    data = dataset.SteelDataset(metadata_root= metadata_path,
                                image_root = images_path,
                                transform = img_t,
                                mask_transform = mask_t)
    matrix_generator = dataset.SteelMatrix(SteelDataset=data,
                                           batch_size=batch_size,
                                           n_workers=num_workers)
    X, M, y = matrix_generator.get_matrix(return_mask = return_mask)
    return X, y, M


def main():
    args = get_argparse()
    X_train, y_train, M_train = create_data(args.image_path, args.train_metadata_path,
                                      batch_size = 200, num_workers=0)
    X_val, y_val, M_val = create_data(args.image_path, args.val_metadata_path,
                                      batch_size = 200, num_workers=0)
    X_test, y_test, M_test = create_data(args.image_path, args.test_metadata_path,
                                      batch_size = 200, num_workers=0)
    torch.save(X_train, args.train_input_path)
    torch.save(y_train, args.train_label_path)
    torch.save(M_train, args.train_mask_path)

    torch.save(X_val, args.val_input_path)
    torch.save(y_val, args.val_label_path)
    torch.save(M_val, args.val_mask_path)

    torch.save(X_test, args.test_input_path)
    torch.save(y_test, args.test_label_path)
    torch.save(M_test, args.test_mask_path)

if __name__ == '__main__':
    main()
