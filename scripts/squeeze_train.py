import sys
import os
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')
from dataset import SteelDataset, Transpose, Resize
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
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve, f1_score


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

    parser.add_argument('--cpu',
                        action = 'store_true')
    parser.add_argument('--n_output',
                        type=int,            default=2)
    parser.add_argument('--batch_size',
                        type=int,            default=10)
    parser.add_argument('--lr',
                        type=float,          default=0.0005)
    parser.add_argument('--patience',
                        type=int,            default=5)
    parser.add_argument('--epochs',
                        type=int,            default=60)
    parser.add_argument('--noise_coeff',
                        type=float,          default=0.15)
    parser.add_argument('--weight_loss',
                        action='store_true')
    parser.add_argument('--save_last',
                        action='store_true')
    args = parser.parse_args()
    return args

def get_model(args, device):
    net = models.squeezenet1_1(pretrained=True)
    net.classifier = nn.Sequential(*net.classifier, nn.Flatten(),
                                   nn.Linear(1000, args.n_output))
    print(net)
    net = net.to(device)
    return net


def get_treshold(t=0.5):
    y_pred_s = y_score.copy()
    y_pred_s[y_score > t] = 0
    y_pred_s[y_score <= t] = 1
    return y_pred_s, f1_score(y_true, y_pred_s, average='macro')

def main():

    ### Set parameters
    args = get_argparse()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    params = get_default_params()
    print('Device: {}'.format(device))
    print('Arguments:\n{}'.format(args))
    print('Parameters:\n{}'.format(params))

    ### Load data
    X_train, y_train= torch.load(args.train_input_path), torch.load(args.train_label_path)
    X_val, y_val = torch.load(args.val_input_path), torch.load(args.val_label_path)

    ### Load model
    net = get_model(args, device)

    ### Train model
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    if args.weight_loss:
        w = [1/y_val[y_val == i].shape[0] for i in range(args.n_output)]
        loss_fn = nn.CrossEntropyLoss(torch.tensor(w).to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.2)

    train_loss_curve, valid_loss_curve = [], []
    train_accuracy, valid_accuracy = [], []
    train_f1, valid_f1 = [], []

    best_loss = np.inf
    patience = 0
    max_patience = args.patience
    bs = args.batch_size

    for ep in range(args.epochs):
        batch_train_loss, batch_valid_loss = [], []
        batch_train_acc, batch_valid_acc = [], []
        batch_train_f1, batch_valid_f1 = [], []

        time.sleep(0.1)
        net.train()
        for n in tqdm(range(X_train.shape[0]//bs)):
            optimizer.zero_grad()
            X, y = X_train[n*bs:(n+1)*bs].to(device), y_train[n*bs:(n+1)*bs].to(device).long()
            X_aug = torch.cat([X, X + torch.randn(*list(X.shape)).to(device)*args.noise_coeff])
            y = torch.cat([y, y])
            out = net(X_aug)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            y_pred = np.argmax(out.detach().cpu().numpy(), axis=1)
            y_true = y.cpu().numpy()
            batch_train_loss.append(float(loss))
            batch_train_acc.append(float(accuracy_score(y_true, y_pred)))
            batch_train_f1.append(float(f1_score(y_true, y_pred, average='macro')))

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
                batch_valid_f1.append(float(f1_score(y.cpu().numpy(), y_pred, average='weighted')))

        lr_scheduler.step()

        train_loss_curve.append(np.mean(batch_train_loss))
        train_accuracy.append(np.mean(batch_train_acc))
        train_f1.append(np.mean(batch_train_f1))

        valid_loss_curve.append(np.mean(batch_valid_loss))
        valid_accuracy.append(np.mean(batch_valid_acc))
        valid_f1.append(np.mean(batch_valid_f1))

        if valid_loss_curve[-1] < best_loss:
            best_loss = valid_loss_curve[-1]
            print('New best parameters have been found. Saving them in {}'.format(args.model_output_path))
            torch.save(net.state_dict(), args.model_output_path)
            patience = 0
        else:
            patience += 1
            time.sleep(0.1)
        print('Epoch: {}'.format(ep + 1))
        print('Train:\tCrossEntropyLoss: {:.5f}\tAccuracy: {:.4f}'.format(train_loss_curve[-1], train_accuracy[-1]), end = '\t')
        print('F1 Score:\t{:.4f}'.format(train_f1[-1]))
        print('Valid:\tCrossEntropyLoss: {:.5f}\tAccuracy: {:.4f}'.format(valid_loss_curve[-1], valid_accuracy[-1]), end = '\t')
        print('F1 Score:\t{:.4f}'.format(valid_f1[-1]))
        time.sleep(0.1)

        if patience >= max_patience:
            print('Max patience has been reached. Stopping training...')
            break


    ### Evaluate model

    # fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    # ax[0].plot(train_loss_curve, label='Train Loss')
    # ax[0].plot(valid_loss_curve, label='Validation Loss')
    # ax[0].legend()
    # ax[0].grid()
    # ax[0].set_title('Loss curve')
    # ax[0].set_xlabel('Epoch')
    # ax[0].set_ylabel('Loss')
    # ax[1].plot(train_accuracy, label='Train Accuracy')
    # ax[1].plot(valid_accuracy, label='Validation Accuracy')
    # ax[1].legend()
    # ax[1].grid()
    # ax[1].set_title('Accuracy curve')
    # ax[1].set_xlabel('Epoch')
    # ax[1].set_ylabel('Accuracy')
    # plt.show()
    #
    # y_pred = []
    # y_true = []
    # y_score = []
    # batch_size = 10
    # with torch.no_grad():
    #     net.eval()
    #     for n in tqdm(range(X_val.shape[0]//batch_size)):
    #         X, y = X_val[n*batch_size:(n+1)*batch_size].to(device), y_val[n*batch_size:(n+1)*batch_size].to(device).long()
    #         out = net(X)
    #         y_score = np.concatenate([y_score, nn.Softmax(dim=1)(out).detach().cpu().numpy().reshape(-1)])
    #         y_pred = np.concatenate([y_pred, np.argmax(out.detach().cpu().numpy(), axis=1)])
    #         y_true = np.concatenate([y_true, y.cpu().numpy()])
    #
    # tpr, fpr, threshold = roc_curve(y_true, y_score)
    # auc_score = auc(fpr, tpr)
    # print('METRICS WITH 0.5 AS THRESHOLD')
    # print('-----------------------------')
    # print('Accuracy:\t{:.4f}'.format(accuracy_score(y_true, y_pred)))
    # print('F1 Score:\t{:.4f}'.format(f1_score(y_true, y_pred, average='macro')))
    # print('AUC Score:\t{:.4f}'.format(auc_score))
    #
    #
    #
    # best_treshold = np.argmax([get_treshold(t=i)[1] for i in np.arange(0, 1, step=0.05)])*0.05
    # y_pred_best = get_treshold(t=best_treshold)[0]
    #
    # accuracy = accuracy_score(y_true, y_pred_best)
    # f1 = f1_score(y_true, y_pred_best, average='macro')
    # print('METRICS WITH {:.2f} THRESHOLD'.format(best_treshold))
    # print('-----------------------------')
    # print('Accuracy:\t{:.4f}'.format(accuracy))
    # print('F1 Score:\t{:.4f}'.format(f1))
    # print('AUC Score:\t{:.4f}'.format(auc_score))
    #
    # ax[0].plot(fpr, tpr, label='AUC = {:.3f}'.format(auc_score))
    # ax[0].grid()
    # ax[0].legend()
    # ax[0].set_title('ROC Curve')
    #
    # cm = confusion_matrix(y_pred_best, y_true)
    # cm_plot = ax[1].matshow(cm, cmap='Blues_r')
    # ax[1].set_title('Confusion Matrix')
    # ax[1].set_xlabel('True')
    # ax[1].set_ylabel('Predicted')
    # ax[1].xaxis.set_ticks_position('bottom')
    # plt.colorbar(cm_plot)
    # ax[1].set_xticklabels(['No Defects', 'No Defects', 'Defects'])
    # ax[1].set_yticklabels(['No Defects', 'No Defects', 'Defects'])
    # for i in range(n_output):
    #     for j in range(n_output):
    #         k = 0
    #         ax[1].text(j, i, cm[i, j], va='center', ha='center')

if __name__ == '__main__':
    main()
