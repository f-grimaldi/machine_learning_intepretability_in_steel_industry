import sys
import os
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')
import models as c_models
from dataset import SlidingWindow
from utils import get_default_params

import numpy as np
import matplotlib.pyplot as plt

import torch
import time
import argparse

from PIL import Image
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import dataloader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.utils import shuffle

def get_argparse():
    # 1. Set argument
    parser = argparse.ArgumentParser(description='Explain SteelNetClassifier')
    # 1.a) General args
    parser.add_argument('--model_output_path', type=str)
    parser.add_argument('--train_input_path',
                        type=str,            default='../data/binaryData/X_train_binary.pth')
    parser.add_argument('--train_label_path',
                        type=str,            default='../data/binaryData/y_train_binary.pth')
    parser.add_argument('--val_input_path',
                        type=str,            default='../data/binaryData/X_val_binary.pth')
    parser.add_argument('--val_label_path',
                        type=str,            default='../data/binaryData/y_val_binary.pth')
    parser.add_argument('--model',
                        type=str,            default='squeeze')
    parser.add_argument('--cpu',
                        action = 'store_true')
    parser.add_argument('--reduced',
                        type=int,            default=-1)
    parser.add_argument('--n_output',
                        type=int,            default=2)
    parser.add_argument('--batch_size',
                        type=int,            default=10)
    parser.add_argument('--lr',
                        type=float,          default=0.0001)
    parser.add_argument('--patience',
                        type=int,            default=5)
    parser.add_argument('--epochs',
                        type=int,            default=60)
    parser.add_argument('--noise_coeff',
                        type=float,          default=0.15)
    parser.add_argument('--save_last',
                        action='store_true')
    parser.add_argument('--vanilla',
                        action='store_true')

    args = parser.parse_args()
    return args

def get_model(args, device):
    if args.model == 'squeeze':
        return c_models.get_squeeze(args, device)
    elif args.model == 'vgg':
        return c_models.get_vgg(args, device)
    elif args.model == 'custom':
        return  c_models.get_cnn(args, device)
    else:
        raise NotImplementedError


def use_noise(X, y, noise_coeff, device):
    if noise_coeff == 0:
        return X, y
    else:
        X = torch.cat([X, X + torch.randn(*list(X.shape)).to(device)*noise_coeff])
        y = torch.cat([y, y])
        return X, y

def get_treshold(t=0.5):
    y_pred_s = y_score.copy()
    y_pred_s[y_score > t] = 0
    y_pred_s[y_score <= t] = 1
    return y_pred_s, f1_score(y_true, y_pred_s, average='macro')

def main():

    ### 1. Set parameters
    args = get_argparse()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    params = get_default_params()
    print('Device: {}'.format(device))
    print('Arguments:\n{}'.format(args))
    print('Parameters:\n{}'.format(params))

    torch.save(torch.tensor([1]), args.model_output_path)
    ### 2. Load data
    X_train = torch.load(args.train_input_path)
    y_train = torch.load(args.train_label_path)

    X_val= torch.load(args.val_input_path)
    y_val = torch.load(args.val_label_path)


    ### 3. Load model
    net = get_model(args, device)

    ### 4. Train model
    ### 4.a Define Optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    ### 4.b loss
    loss_fn = nn.CrossEntropyLoss()
    ### 4.c Set scheduler
    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.2)

    ### 4.d Init training variables
    train_loss_curve, valid_loss_curve = [], []
    train_accuracy, valid_accuracy = [], []
    train_f1, valid_f1 = [], []
    train_bal_accuracy, valid_bal_accuracy = [], []

    best_acc = 0
    patience = 0
    max_patience = args.patience
    bs = args.batch_size

    ### 4.e Run an epoch
    for ep in range(args.epochs):
        batch_train_loss, batch_valid_loss = [], []
        batch_train_acc, batch_valid_acc = [], []
        batch_train_f1, batch_valid_f1 = [], []
        batch_train_bal_acc, batch_valid_bal_acc = [], []
        time.sleep(0.1)

        ### TRAIN
        net.train()
        for n in tqdm(range(X_train.shape[0]//bs)):
            optimizer.zero_grad()
            X, y = X_train[n*bs:(n+1)*bs].to(device), y_train[n*bs:(n+1)*bs].to(device).long()
            X, y = use_noise(X, y, args.noise_coeff, device)

            #print(X.shape)
            out = net(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            y_pred = np.argmax(out.detach().cpu().numpy(), axis=1)
            y_true = y.cpu().numpy()
            batch_train_loss.append(float(loss))
            batch_train_acc.append(float(accuracy_score(y_true, y_pred)))
            batch_train_bal_acc.append(float(balanced_accuracy_score(y_true, y_pred)))
            batch_train_f1.append(float(f1_score(y_true, y_pred, average='weighted')))

        ### VALIDATION
        time.sleep(0.1)
        with torch.no_grad():
            net.eval()
            for n in tqdm(range(X_val.shape[0]//bs)):
                X, y = X_val[n*bs:(n+1)*bs].to(device), y_val[n*bs:(n+1)*bs].to(device).long()
                out = net(X)
                loss = loss_fn(out, y)

                y_pred = np.argmax(out.detach().cpu().numpy(), axis=1)
                batch_valid_loss.append(float(loss))
                batch_valid_acc.append(float(accuracy_score(y.cpu().numpy(), y_pred)))
                batch_valid_bal_acc.append(float(balanced_accuracy_score(y.cpu().numpy(), y_pred)))
                batch_valid_f1.append(float(f1_score(y.cpu().numpy(), y_pred, average='weighted')))

        #lr_scheduler.step()

        ### Check results
        train_loss_curve.append(np.mean(batch_train_loss))
        train_accuracy.append(np.mean(batch_train_acc))
        train_bal_accuracy.append(np.mean(batch_train_bal_acc))
        train_f1.append(np.mean(batch_train_f1))

        valid_loss_curve.append(np.mean(batch_valid_loss))
        valid_accuracy.append(np.mean(batch_valid_acc))
        valid_f1.append(np.mean(batch_valid_f1))
        valid_bal_accuracy.append(np.mean(batch_valid_bal_acc))

        if valid_accuracy[-1] > best_acc:
            best_acc = valid_accuracy[-1]
            print('New best parameters have been found. Saving them in {}'.format(args.model_output_path))
            torch.save(net.state_dict(), args.model_output_path)
            patience = 0
        else:
            patience += 1
            time.sleep(0.1)
        print('Epoch: {}'.format(ep + 1))
        print('Train:\tCrossEntropyLoss: {:.4f}\tAccuracy: {:.4f}'.format(train_loss_curve[-1], train_accuracy[-1]), end = '\t')
        print('F1 Score: {:.4f}\tBalanced Accuracy:\t{:.4f}'.format(train_f1[-1], train_bal_accuracy[-1]))
        print('Valid:\tCrossEntropyLoss: {:.4f}\tAccuracy: {:.4f}'.format(valid_loss_curve[-1], valid_accuracy[-1]), end = '\t')
        print('F1 Score: {:.4f}\tBalanced Accuracy:\t{:.4f}'.format(valid_f1[-1], valid_bal_accuracy[-1]))
        time.sleep(0.1)

        if patience >= max_patience:
            print('Max patience has been reached. Stopping training...')
            break


    ### 5. Plot loss curves and accuracies

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax[0].plot(train_loss_curve, label='Train Loss')
    ax[0].plot(valid_loss_curve, label='Validation Loss')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_title('Loss curve')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].plot(train_accuracy, label='Train Accuracy')
    ax[1].plot(valid_accuracy, label='Validation Accuracy')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_title('Accuracy curve')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    main()
