import numpy as np
import pandas as pd
import sklearn
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import os
from  sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_recall_curve, auc, recall_score, precision_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.utils import resample
import json
import sys

warnings.filterwarnings('ignore')

class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__() 
        #Defining CNN sequential model
        self.cnn = nn.Sequential( 
                                    nn.Conv1d(2, 8, 3, padding=1), # 1D convolution 
                                    nn.LayerNorm((8, 256)), #Normalisation layer
                                    nn.ReLU(), # Using Relu Activation
                                    nn.MaxPool1d(2), #Applying Max pooling
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


def classification(param, X_matrix, y_vector, report_file):
    '''
    Params:
    param: hyperparameters for model training
    '''
    n_epochs = param["n_epochs"]
    display_epochs = param["display_epochs"]
    lr = param["lr"]
    weight_decay = param["weight_decay"]
    
    #reading images and labels
    # img_file = os.path.join(directory, "data/images.json")
    # lbl_file = os.path.join(directory, "data/labels.json")
    
    #type conversion to numpy arrays
    # X_matrix = np.array(X_matrix) 
    # y_vector = np.array(y_vector)
    # print(X_matrix)
    # print(y_vector)
    # print(X_matrix.shape, y_vector.shape)

    # normalize input data
    # X_matrix = np.array((X_matrix - X_matrix.mean()) / X_matrix.std(), dtype = np.float32)
    # print(X_matrix.size)
   
    # Defining train and test size

    train_size = int(0.6*len(X_matrix))
    val_size = int(0.2*len(X_matrix))
    test_size = int(0.2*len(X_matrix))
    
    X_train = []
    y_train = []
    # train / test split data
    for i in range(train_size):
        X_train.append(X_matrix[i])
        y_train.append(y_vector[i])
    
    #Converting to torch tensors
    X_train = torch.from_numpy(np.array(X_train))
    y_train = torch.from_numpy(np.array(y_train))
    
    X_val = []
    y_val = []
    for i in range(train_size, train_size + val_size):
        X_val.append(X_matrix[i])
        y_val.append(y_vector[i])
    
    X_val = torch.from_numpy(np.array(X_val))
    y_val = torch.from_numpy(np.array(y_val))
    
    X_test = []
    y_test = []
    for i in range(train_size + val_size, train_size + val_size + test_size):
        X_test.append(X_matrix[i])
        y_test.append(y_vector[i])
   
    #defining model metrics and optimiser
    X_test = torch.from_numpy(np.array(X_test))
    y_test = torch.from_numpy(np.array(y_test))
    
    net = Net()
    criterion = nn.BCELoss()#reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay)#optim.SGD(net.parameters(), lr=0.001)#, momentum=0.8)
    epochs = np.arange(n_epochs)

    best_loss_val = float('inf')
    best_state_on_val = None
    
    # Training the model 
    
    for epoch in list(epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0
        net.train()
        for i in range(train_size):
            # get the inputs; info is a list of [inputs, labels]
            inputs = X_train[i]
            inputs = inputs[None, :]
            # print(inputs.shape)
            labels = y_train[i]
            # print(labels)
            # print(labels.shape)
            labels = labels[None]

            # zero the parameter gradients
            for param in net.parameters():
                param.grad = None

            # forward + backward + optimize
            #print(inputs.size())bootstrap_estimate_mean_stddev
            # print(net(inputs).size())
            outputs = net(inputs).reshape(1)#(61)
            # print(outputs.size())
            # print(labels.size())
            outputs = outputs.type(torch.float32)
            labels = labels.type(torch.float32)
            loss = criterion(outputs, labels)
            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # print mean loss for the epoch
        cur_loss = epoch_loss / X_train.shape[0]
        # plt.plot(epoch, cur_loss, '.', color='red')
        if (epoch + 1) % display_epochs == 0:
            print('[%5d] loss: %.3f' % (epoch + 1, cur_loss))
            original_stdout = sys.stdout
            with open(report_file, 'a') as f:
                sys.stdout = f
                print('[%5d] loss: %.3f' % (epoch + 1, cur_loss))
                sys.stdout = original_stdout
        #model evaluation
        net.eval()
        epoch_loss_val = 0.0
        for i in range(val_size):
            # get the inputs; info is a list of [inputs, labels]
            inputs = X_val[i]
            labels = y_val[i]
            inputs = inputs[None, :]
            labels = labels[None]

            # forward
            outputs = net(inputs).reshape(1)
            outputs = outputs.type(torch.float32)
            labels = labels.type(torch.float32)
            loss = criterion(outputs, labels)

            epoch_loss_val += loss.item()

        cur_loss_val = epoch_loss_val / X_val.shape[0]
        # plt.plot(epoch, cur_loss_val, '.', color='blue')

        if epoch_loss_val <= best_loss_val:
            best_loss_val = epoch_loss_val
            best_state_on_val = deepcopy(net.state_dict())

    # plt.legend(['train_loss', 'val_loss'])        
    # plt.show()

    net.load_state_dict(best_state_on_val)
    
    print('Finished Training CNN')
    # X_test = nf.X_test
    # y_test = nf.y_test
    # img_file = os.path.join(directory, "X_test.json")
    # lbl_file = os.path.join(directory, "y_test.json")
    
    # with open(img_file, 'r') as f:
    #     X_test = json.load(f)
    # with open(lbl_file, 'r') as f:
    #     y_test = json.load(f)
        
    # X_test = np.array(X_test)
    # y_test = np.array(y_test)
        
    # X_test = np.array((X_test - X_test.mean()) / X_test.std(), dtype = np.float32)
    
    # X_test = torch.from_numpy(np.array(X_test))
    
    # y_test = torch.from_numpy(np.array(y_test))
    # print(X_test)
    
    # print(y_test)

    y_test_pred = net(X_test).detach().numpy()[:, 0]
    y_test_pred = np.array(y_test_pred)
    # print(y_test_pred)
    
    # original_stdout = sys.stdout
    # with open(report_file, 'a') as f:
    #     sys.stdout = f
    #     print(y_test)
    #     print(y_test_pred)
    #     sys.stdout = original_stdout
    
    # y_test_pred_list = y_test_pred.tolist()
    #with open(directory + "/y_test_pred.json", 'w') as f:
        #json.dump(y_test_pred_list, f)
    return y_test, y_test_pred
