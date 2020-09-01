
if __name__ == '__main__':
    import sys
    import os
    sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')
    from dataset import SteelDataset, Transpose, Resize
    from utils import get_default_params

    import numpy as np
    import matplotlib.pyplot as plt

    import torch
    import time

    from PIL import Image
    from tqdm import tqdm
    from torch import nn, optim
    from torch.utils.data import dataloader
    from torchvision import transforms, models
    from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve, f1_score

    ### PARAMS
    BASE_PATH = '../data/binaryData'
    TRAIN_INPUT_PATH, TRAIN_LABEL_PATH = '/X_train_binary.pth', '/y_train_binary.pth'
    VAL_INPUT_PATH, VAL_LABEL_PATH = '/X_val_binary.pth', '/y_val_binary.pth'
    SAVE_PATH = '../model/SqueezeBinary101.pth'

    device = torch.device('cuda')
    params = get_default_params()
    print('Parameters: {}'.format(params))

    ### DATA
    X_train = torch.load(BASE_PATH+ '/X_train_binary.pth')
    y_train = torch.load(BASE_PATH+ '/y_train_binary.pth')
    X_val = torch.load(BASE_PATH+ '/X_val_binary.pth')
    y_val = torch.load(BASE_PATH+ '/y_val_binary.pth')
    print('Train shape: {}; {}'.format(X_train.shape, y_train.shape))
    print('Valid shape: {}; {}'.format(X_val.shape, y_val.shape))

    n=0
    img = np.transpose(X_train[n].numpy(), (1, 2, 0))*params['std'] + params['mean']
    plt.imshow(img)
    plt.title('True label: {}'.format(int(y_train[n])))
    plt.show()

    ### MODEL
    net = models.squeezenet1_1(pretrained=True)
    net.classifer = nn.Sequential(*net.classifier, nn.Flatten(),
                                  nn.Linear(1000, 32), nn.ReLU(), nn.Dropout(p=0.3),
                                  nn.Linear(32, 2))
    net = net.to(device)

    ### TRAINING
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.2)

    is_trained = False
    batch_size = 10

    train_loss_curve, valid_loss_curve = [], []
    train_accuracy, valid_accuracy = [], []
    train_f1, valid_f1 = [], []

    best_loss = np.inf
    patience = 0
    max_patience = 5

    for ep in range(60):
        batch_train_loss, batch_valid_loss = [], []
        batch_train_acc, batch_valid_acc = [], []
        batch_train_f1, batch_valid_f1 = [], []

        time.sleep(0.1)
        net.train()
        for n in tqdm(range(X_train.shape[0]//batch_size)):
            optimizer.zero_grad()
            X, y = X_train[n*batch_size:(n+1)*batch_size].to(device), y_train[n*batch_size:(n+1)*batch_size].to(device).long()
            X_aug = torch.cat([X, X + torch.randn(*list(X.shape)).to(device)*0.15])
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
            for n in tqdm(range(X_val.shape[0]//batch_size)):
                X, y = X_val[n*batch_size:(n+1)*batch_size].to(device), y_val[n*batch_size:(n+1)*batch_size].to(device).long()
                out = net(X)
                loss = loss_fn(out, y)

                y_pred = np.argmax(out.detach().cpu().numpy(), axis=1)
                batch_valid_loss.append(float(loss))
                batch_valid_acc.append(float(accuracy_score(y.cpu().numpy(), y_pred)))
                batch_valid_f1.append(float(f1_score(y.cpu().numpy(), y_pred, average='macro')))

        lr_scheduler.step()

        train_loss_curve.append(np.mean(batch_train_loss))
        train_accuracy.append(np.mean(batch_train_acc))
        train_f1.append(np.mean(batch_train_f1))

        valid_loss_curve.append(np.mean(batch_valid_loss))
        valid_accuracy.append(np.mean(batch_valid_acc))
        valid_f1.append(np.mean(batch_valid_f1))

        if valid_loss_curve[-1] < best_loss:
            best_loss = valid_loss_curve[-1]
            print('New best parameters have been found. Saving them in {}'.format(SAVE_PATH))
            torch.save(net.state_dict(), SAVE_PATH)
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

    ###EVAL

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

    y_pred = []
    y_true = []
    y_score = []
    batch_size = 10
    with torch.no_grad():
        net.eval()
        for n in tqdm(range(X_val.shape[0]//batch_size)):
            X, y = X_val[n*batch_size:(n+1)*batch_size].to(device), y_val[n*batch_size:(n+1)*batch_size].to(device).long()
            out = net(X)
            y_score = np.concatenate([y_score, nn.Softmax(dim=1)(out).detach().cpu().numpy().reshape(-1)])
            y_pred = np.concatenate([y_pred, np.argmax(out.detach().cpu().numpy(), axis=1)])
            y_true = np.concatenate([y_true, y.cpu().numpy()])
    print('Accuracy: {}'.format(accuracy_score(y_true, y_pred)))
    print('F1 Score: {}'.format(f1_score(y_true, y_pred, average='macro')))

    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    cm = confusion_matrix(y_pred, y_true)
    cm_plot = ax.matshow(cm, cmap='Blues_r')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    plt.colorbar(cm_plot)
    for i in range(2):
        for j in range(2):
            k = 0
            ax.text(j, i, cm[i, j], va='center', ha='center')
