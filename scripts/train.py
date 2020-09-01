import numpy as np
import matplotlib.pyplot as plt

import torch
import os
import time

from PIL import Image
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import dataloader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve, f1_score

from src.utils import get_default_params
from src.dataset import SteelDataset, Transpose, Resize

BASE_PATH = '../data/BinaryTask'
TRAIN_INPUT_PATH, TRAIN_LABEL_PATH = '/X_train_binary.pth', '/y_train_binary.pth'
VAL_INPUT_PATH, VAL_LABEL_PATH = '/X_val_binary.pth', '/y_val_binary.pth'


X_train, y_train = torch.load(BASE_PATH+ '/X_train_binary.pth'), torch.load(BASE_PATH+ '/y_train_binary.pth')
X_val, y_val = torch.load(BASE_PATH+ '/X_val_binary.pth'), torch.load(BASE_PATH+ '/y_val_binary.pth')

print('Train shape: {}; {}'.format(X_train.shape, y_train.shape))
print('Train mean and std: {}; {}'.format(X_train.mean(), X_train.std()))
print('Valid shape: {}; {}'.format(X_val.shape, y_val.shape))
print('Valid mean and std: {}; {}'.format(X_val.mean(), X_val.std()))

params = get_default_params()
