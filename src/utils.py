from matplotlib import colors
import numpy as np
import pandas as pd
import json
import shutil
import os
from sklearn.utils import shuffle
try:
    import src.dataset as dataset
except:
    import dataset
import torch
from torch import nn
from torchvision import transforms

def mask_metrics(m_true, mask, cutoff=0.7):
    m_pred = mask.clone()
    m_pred[m_pred < 0.7] = 0
    m_pred[m_pred >= 0.7] = 1
    m_pred = m_pred.cpu().long()

    tp = m_pred[(m_true == 1) & (m_pred == 1)].shape[0]
    fp = m_pred[(m_true == 0) & (m_pred == 1)].shape[0]
    tn = m_pred[(m_true == 0) & (m_pred == 0)].shape[0]
    fn = m_pred[(m_true == 1) & (m_pred == 0)].shape[0]

    return m_pred, {'Accuracy': (tp+tn)/(tp+fp+fn+tn),
                    'Recall':tp/(tp+fn),
                    'Dice': tp/(tp+fp)}


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [self.vmin, self.vcenter, self.vmax]
        return np.ma.masked_array(np.interp(value, x, y))

def train_val_split(metadata_root, val_size, shuffle_data=False):
    df = pd.read_csv(metadata_root)
    val_index = np.random.choice(np.arange(0, df.shape[0]), int(val_size*df.shape[0]), replace=False)
    df_train = df.iloc[df.index.isin(val_index) == False, :]
    df_val = df.iloc[val_index, :]
    if shuffle_data:
        return shuffle(df_train), shuffle(df_val)
    return df_train, df_val

def train_val_test_split(df, val_size, test_size, shuffle_data=True):
    val_index = np.random.choice(np.arange(0, df.shape[0]), int(val_size*df.shape[0]), replace=False)
    df_train = df.iloc[df.index.isin(val_index) == False, :]
    df_val_test = df.iloc[val_index, :]
    test_index = np.random.choice(df_val_test.index, int(test_size*df_val_test.shape[0]), replace=False)

    df_val = df_val_test.iloc[df_val_test.index.isin(test_index) == False, :]
    df_test = df_val_test.iloc[df_val_test.index.isin(test_index), :]
    if shuffle_data:
        return shuffle(df_train), shuffle(df_val), shuffle(df_test)
    return df_train, df_val, df_test

### Get unqiue metadata
def get_unique_metadata(METADATA_PATH, max_imgs=40):
    metadata = pd.read_csv(METADATA_PATH)
    unique_imgs = []
    unique_index = []
    for i in tqdm(range(metadata.shape[0])):
        if metadata.ImageId[i] not in unique_imgs:
            unique_imgs.append(metadata.ImageId[i])
            unique_index.append(metadata.index[i])
    ### Add Folder information to the dataframe
    unique_metadata = metadata[metadata.index.isin(unique_index)]
    unique_metadata.index = np.arange(0, unique_metadata.shape[0])
    unique_metadata['Folder'] = np.arange(0, unique_metadata.shape[0])//max_imgs
    return unique_metadata

### Create folder and move the imgs
def chunck(unique_metadata, IMG_PATH):
    for i in tqdm(unique_metadata.Folder.unique()):
        if str(i) not in os.listdir(IMG_PATH):
            os.mkdir(IMG_PATH +'/' + str(i))
            tmp_imgs = unique_metadata.ImageId[unique_metadata.Folder == i]
            for img in tmp_imgs:
                shutil.move(IMG_PATH + '/' + img, IMG_PATH +'/' + str(i))
        else:
            print(i, end=', ')

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

def get_transform(size, mean, std):
    img_transform = transforms.Compose([transforms.Resize(size = size), transforms.ToTensor(),
                                      transforms.Normalize(mean = mean, std = std)])
    mask_transform = transforms.Compose([dataset.Resize(size = (size[1], size[0]))])
    return img_transform, mask_transform


def get_default_params():
    params = {'size': (64, 400),
              'mask_size': (64, 400),
              'mean': [0.485, 0.456, 0.406],
              'std': [0.229, 0.224, 0.225],
              'colors': [[0, 0.25, 0.25], [0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]}
    return params

def vgg_load_state_dict(net, JSON_PATH, verbose=0):
    ### 4.2 Load Parameters
    with open(JSON_PATH, 'r') as f:
        params_data = json.load(f)
    if verbose:
        print(params_data.keys())
    for key in params_data.keys():
        module, layer, type_ = key.split('.')
        if module == 'features':
            if type_ == 'weight':
                #print('Feature weight.\tTrue:\t{}\tLoaded: {}, {}, {}'.format(net.features[int(layer)], module, layer, type_))
                net.features[int(layer)].weight = nn.Parameter(torch.tensor(params_data[key]))
            elif type_ == 'bias':
                #print('Feature bias.\tTrue:\t{}\tLoaded: {}, {}, {}'.format(net.features[int(layer)], module, layer, type_))
                net.features[int(layer)].bias = nn.Parameter(torch.tensor(params_data[key]))
            elif type_ == 'running_mean':
                net.features[int(layer)].running_mean = nn.Parameter(torch.tensor(params_data[key]))
                net.features[int(layer)].running_mean.requires_grad = False
            elif type_ == 'running_var':
                net.features[int(layer)].running_var = nn.Parameter(torch.tensor(params_data[key]))
                net.features[int(layer)].running_var.requires_grad = False
            else:
                net.features[int(layer)].num_batches_tracked = torch.tensor(params_data[key])
        else:
            if type_ == 'weight':
                net.classifier[int(layer)].weight = nn.Parameter(torch.tensor(params_data[key]))
            else:
                net.classifier[int(layer)].bias = nn.Parameter(torch.tensor(params_data[key]))
    return net
