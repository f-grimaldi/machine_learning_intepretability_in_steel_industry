import torch
import numpy as np
from tqdm import tqdm
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from torch import nn

"""
SQUEEZE
"""
def get_squeeze(args, device):
    if not args.vanilla:
        net = models.squeezenet1_1(pretrained=True)
    else:
        net = models.squeezenet1_1(pretrained=False)
    if args.reduced > 0:
        filters = {11: 384, 10: 384, 9:256, 8:256, 7:256}
        net.features = net.features[:args.reduced]
        net.classifier[1] = nn.Conv2d(filters[args.reduced],
                                      1000,
                                      kernel_size = (1, 1),
                                      stride = (1, 1))

    net.classifier = nn.Sequential(*net.classifier, nn.Flatten(), nn.Linear(1000, args.n_output))


    print(net)
    net = net.to(device)
    return net

"""
VGG
"""
def get_vgg(args, device):
    if not args.vanilla:
        net = models.vgg16_bn(pretrained=True)
    else:
        net = models.vgg16_bn(pretrained=False)

    filters = {-1: 25088, 10: 384, 9:256, 8:256, 7:256}
    net.classifier = nn.Sequential(nn.Linear(25088, 256), nn.ReLU(), nn.Dropout(0.5),
                                   nn.Linear(256, 32), nn.ReLU(), nn.Dropout(0.5),
                                   nn.Linear(32, args.n_output))
    print(net)
    net = net.to(device)
    return net

"""
AD_HOC
"""
class CNN(nn.Module):
    def __init__(self, n_output=5):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(3, 3)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(2, 2)),
                                      nn.Conv2d(32, 32, kernel_size=(3, 3)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(2, 2)),
                                      nn.Conv2d(32, 32, kernel_size=(3, 3)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(2, 2)),
                                      nn.Conv2d(32, 32, kernel_size=(3, 3)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(2, 2)))
        self.linear = nn.Sequential(nn.Linear(32*2*23, 256),
                                    nn.Dropout(p=0.3),
                                    nn.ReLU(),
                                    nn.Linear(256, 32),
                                    nn.Dropout(p=0.3),
                                    nn.ReLU(),
                                    nn.Linear(32, n_output))
    def forward(self, x):
        features = self.features(x).view(x.shape[0], -1)
        return self.linear(features)

def get_cnn(args, device):
    return CNN(args.n_output).to(device)

def use_noise(X, y, noise_coeff, device):
    if noise_coeff == 0:
        return X, y
    else:
        X = torch.cat([X, X + torch.randn(*list(X.shape)).to(device)*noise_coeff])
        y = torch.cat([y, y])
        return X, y

def fit(net,
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size,
        epochs,
        loss,
        optim,
        patience,
        noise=0.15,
        device = torch.device('cuda'),
        tqdm_disable=False):

    raise NotImplementedError

def train_step(net, X, y, bs,
               loss_fn,
               optimizer,
               noise = 0.15,
               tqdm_disable=False,
               device = torch.device('cuda')):

    batch_loss =  []
    batch_acc = []
    batch_f1 = []
    batch_bal_acc = []
    net.train()
    for n in tqdm(range(X.shape[0]//bs), disable = tqdm_disable):
        optimizer.zero_grad()
        X, y = X[n*bs:(n+1)*bs].to(device), y[n*bs:(n+1)*bs].to(device).long()
        X, y = use_noise(X, y, noise, device)

        out = net(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        y_pred = np.argmax(out.detach().cpu().numpy(), axis=1)
        y_true = y.cpu().numpy()
        batch_loss.append(float(loss))
        batch_acc.append(float(accuracy_score(y_true, y_pred)))
        batch_bal_acc.append(float(balanced_accuracy_score(y_true, y_pred)))
        batch_f1.append(float(f1_score(y_true, y_pred, average='weighted')))

    return {'loss': np.mean(batch_loss),
            'accuracy': np.mean(batch_acc),
            'bal_accuracy': np.mean(batch_bal_acc),
            'f1_score': np.mean(batch_f1)}

def eval_step(net, X, y, bs,
              loss,
              tqdm_disable = False,
              device = torch.device('cuda')):
    batch_loss =  []
    batch_acc = []
    batch_f1 = []
    batch_bal_acc = []
    with torch.no_grad():
        net.eval()
        for n in tqdm(range(X.shape[0]//bs), disable = tqdm_disable):
            optimizer.zero_grad()
            X, y = X[n*bs:(n+1)*bs].to(device), y[n*bs:(n+1)*bs].to(device).long()

            out = net(X)
            loss = loss_fn(out, y)

            y_pred = np.argmax(out.detach().cpu().numpy(), axis=1)
            y_true = y.cpu().numpy()
            batch_loss.append(float(loss))
            batch_acc.append(float(accuracy_score(y_true, y_pred)))
            batch_bal_acc.append(float(balanced_accuracy_score(y_true, y_pred)))
            batch_f1.append(float(f1_score(y_true, y_pred, average='weighted')))

    return {'loss': np.mean(batch_loss),
            'accuracy': np.mean(batch_acc),
            'bal_accuracy': np.mean(batch_bal_acc),
            'f1_score': np.mean(batch_f1)}
