import numpy as np
import pandas as pd
import sklearn
import warnings
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.gaussian_process.kernels import RBF, Matern, \
RationalQuadratic, WhiteKernel, DotProduct, ConstantKernel as C
from importlib import reload
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import fulu
from fulu import single_layer_aug
from fulu import bnn_aug
from fulu import nf_aug
from fulu import mlp_reg_aug
from fulu import gp_aug
#import gp_aug as gp_aug_old
#from gp_aug import bootstrap_estimate_mean_stddev
from copy import deepcopy
import os
from joblib import Parallel, delayed
from  sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_recall_curve, auc, recall_score, precision_score
from sklearn.utils import resample
import json
warnings.filterwarnings('ignore')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
                                    nn.Conv1d(2, 8, 3, padding=1),
                                    nn.LayerNorm((8, 256)),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2),
                                    nn.Conv1d(8, 16, 3, padding=1),
                                    nn.LayerNorm((16, 128)),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2),
                                    nn.Conv1d(16, 32, 3, padding=1),
                                    nn.LayerNorm((32, 64)),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2),
                                    nn.Flatten(),
                                    nn.Dropout(0.33),
                                    nn.Linear(16 * 64, 1),
                                    nn.Sigmoid()
                                )

    def forward(self, x):
        x = self.cnn(x)
        return x

def classification(n_epoches = 10):
    all_data = []
    all_target_classes = []

    directory = os.path.dirname(__file__)
    with open(directory + "/images.json", 'r') as f:
        all_data = json.load(f)
    with open(directory + "/labels.json", 'r') as f:
        all_target_classes = json.load(f)
    
    all_data = np.array(all_data)
    all_target_classes = np.array(all_target_classes)
    print(all_data.shape, all_target_classes.shape)

    # train / test split data
    X_train, X_test_val, y_train, y_test_val = train_test_split(all_data, 
                                                            all_target_classes,
                                                            test_size=0.4,
                                                            random_state=11)

    X_val, X_test, y_val, y_test = train_test_split(X_test_val, 
                                                            y_test_val,
                                                            test_size=0.5,
                                                            random_state=11)
    # normalize input data
    X_train_norm = np.array((X_train - X_train.mean()) / X_train.std(), dtype=np.float32)
    X_test_norm = np.array((X_test - X_train.mean()) / X_train.std(), dtype=np.float32)
    X_val_norm = np.array((X_val - X_train.mean()) / X_train.std(), dtype=np.float32)

    # convert train data to tensors
    X_train_tensor = torch.from_numpy(X_train_norm)
    y_train_tensor = torch.from_numpy(np.array(y_train, dtype=np.float32))

    # create train data loader
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=1,
                                                  shuffle=True)#, num_workers=2)

    # convert test data to tensors
    X_test_tensor = torch.from_numpy(X_test_norm)
    y_test_tensor = torch.from_numpy(np.array(y_test, dtype=np.float32))

    # create test data loader
    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                 shuffle=False)#, num_workers=2)

    # convert val data to tensors
    X_val_tensor = torch.from_numpy(X_val_norm)
    y_val_tensor = torch.from_numpy(np.array(y_val, dtype=np.float32))

    # create val data loader
    val_data = TensorDataset(X_val_tensor, y_val_tensor)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=1,
                                                 shuffle=False)#, num_workers=2)
    
    #print(X_train_tensor.size())
    net = Net()
    criterion = nn.BCELoss()#reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=0.0002, weight_decay=0.001)#optim.SGD(net.parameters(), lr=0.001)#, momentum=0.8)
    epochs = np.arange(n_epoches)

    best_loss_val = float('inf')
    best_state_on_val = None

    for epoch in list(epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0
        net.train()
        for info in trainloader:
            # get the inputs; info is a list of [inputs, labels]
            inputs, labels = info

            # zero the parameter gradients
            for param in net.parameters():
                param.grad = None

            # forward + backward + optimize
            #print(inputs.size())bootstrap_estimate_mean_stddev
            #print(net(inputs).size())
            outputs = net(inputs).reshape(1)#(61)
            #print(outputs.size())
            #print(labels.size())
            loss = criterion(outputs, labels)
            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # print mean loss for the epoch
        cur_loss = epoch_loss / X_train_norm.shape[0]
        # plt.plot(epoch, cur_loss, '.', color='red')
        if (epoch + 1) % 10 == 0:
            print('[%5d] error: %.3f' % (epoch + 1, cur_loss))

        net.eval()
        epoch_loss_val = 0.0
        for info in valloader:
            # get the inputs; info is a list of [inputs, labels]
            inputs, labels = info

            # forward
            outputs = net(inputs).reshape(1)
            loss = criterion(outputs, labels)

            epoch_loss_val += loss.item()

        cur_loss_val = epoch_loss_val / X_val_norm.shape[0]
        # plt.plot(epoch, cur_loss_val, '.', color='blue')

        if epoch_loss_val <= best_loss_val:
            best_loss_val = epoch_loss_val
            best_state_on_val = deepcopy(net.state_dict())

    # plt.legend(['train_loss', 'val_loss'])        
    # plt.show()

    net.load_state_dict(best_state_on_val)
    
    print('Finished Training')
