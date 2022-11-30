import numpy as np
import pandas as pd
import sklearn
import warnings
warnings.filterwarnings('ignore')
# import utils
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
from  sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_recall_curve, auc, recall_score, precision_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.utils import resample
import json

path = os.getcwd()
path_to = "{}/data/ZTF_BTS_23_29__22_09_2021.csv".format(os.getcwd())

bts = pd.read_csv(path_to, sep =',')
bts = bts.drop('Unnamed: 0', 1)

df_all = pd.read_csv(path + '/data/ANTARES_NEW.csv')
#df_all = pd.read_csv(path + 'ANTARES_10_in_g_r_bands.csv')
df_all = df_all.drop('Unnamed: 0', 1)

print("Названия колонок в таблице ANTARES_NEW.csv со всеми кривыми блеска: \n\n", df_all.columns, "\n\n")
print("Number of objects: ", len(df_all['object_id'].unique()))

obj_names = df_all['object_id'].unique()

# df_all.loc[df_all.obj_type == 'SN Ia', 'obj_type'] = 1
# df_all.loc[df_all.obj_type != 1, 'obj_type'] = 0

df_all.loc[df_all.obj_type == 'SN Ia', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Ia-91T', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Ia-pec', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Iax', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Ia-91bg', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Ia-CSM', 'obj_type'] = 1
df_all.loc[df_all.obj_type != 1, 'obj_type'] = 0 

def bootstrap_estimate_mean_stddev(arr, n_samples=10000):
    arr = np.array(arr)
    np.random.seed(0)
    bs_samples = np.random.randint(0, len(arr), size=(n_samples, len(arr)))
    bs_samples = arr[bs_samples].mean(axis=1)
    sigma = np.sqrt(np.sum((bs_samples - bs_samples.mean())**2) / (n_samples - 1))
    return np.mean(bs_samples), sigma

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

def gen_report(y_test, y_test_pred, n_iters=1000, decimals=3):
    
    metrics = []
    inds = np.arange(len(y_test))
    for i in range(n_iters):
        inds_boot = resample(inds)
        roc_auc = roc_auc_score(y_test[inds_boot], y_test_pred[inds_boot])
        logloss = log_loss(y_test[inds_boot], y_test_pred[inds_boot], eps=10**-6)
        accuracy = accuracy_score(y_test[inds_boot], 1 * (y_test_pred[inds_boot] > 0.5))
        precision, recall, _ = precision_recall_curve(y_test[inds_boot], y_test_pred[inds_boot])
        pr_auc = auc(recall, precision)
        recall = recall_score(y_test[inds_boot], 1 * (y_test_pred[inds_boot] > 0.5))
        precision = precision_score(y_test[inds_boot], 1 * (y_test_pred[inds_boot] > 0.5))
        RMSE = mean_squared_error(y_test[inds_boot], y_test_pred[inds_boot], squared=False)             # Mean Sqaure Error
        MAE = mean_absolute_error(y_test[inds_boot], y_test_pred[inds_boot]) 
        MAPE = mean_absolute_percentage_error(y_test[inds_boot], y_test_pred[inds_boot])
        metrics.append([roc_auc, pr_auc, logloss, accuracy, recall, precision, RMSE, MAE, MAPE])
    metrics = np.array(metrics)
    report = pd.DataFrame(columns=["ROC_AUC", 'PR-AUC', 'LogLoss', 'Accuracy', 'Recall', 'Precision', 'RMSE', 'MAE', 'MAPE'],
                          data=[metrics.mean(axis=0), metrics.std(axis=0)], 
                          index=['mean', 'std'])
    
    return report

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
            print(labels)
            print(labels.shape)

            # zero the parameter gradients
            for param in net.parameters():
                param.grad = None

            # forward + backward + optimize
            #print(inputs.size())bootstrap_estimate_mean_stddev
            print(net(inputs).shape)
            outputs = net(inputs).reshape(1)#(61)
            print(outputs.shape)
            print(labels.shape)
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
    
    y_test_pred = net(X_test_tensor).detach().numpy()[:, 0]
    
    report = gen_report(y_test, y_test_pred)
    print(report)
    # y_test = []
    # y_probs = []
    # y_probs_0 = []
    # y_probs_1 = []

    # with torch.no_grad():
    #     for test_info in testloader:
    #         images, test_labels = test_info
    #         test_outputs = net(images)

    #         # get output value
    #         prob = test_outputs.item()

    #         # check true target valur    
    #         true_class = int(test_labels.item())

    #         # compare output to threshold
    #         if true_class == 0:
    #             y_probs_0.append(prob)
    #         else:
    #             y_probs_1.append(prob)

    #         # get predicted target value
    #         y_test.append(true_class)
    #         y_probs.append(prob)

    # y_test = np.array(y_test)
    # y_probs = np.array(y_probs)

    # assert np.array(y_probs).min() >= 0
    # assert np.array(y_probs).max() <= 1

    # N = len(y_probs)

    # # sample predicted values
    # sample_coeffs = np.random.randint(0, N, (10000, 1000))
    # sample_prob = y_probs[sample_coeffs]
    # sample_test = y_test[sample_coeffs]
    # sample_pred = sample_prob > 0.5

    # assert len(sample_test) == len(sample_prob)
    # assert len(sample_prob) == len(sample_pred)
    # T = len(sample_test)

    # # calculated mean accuracy
    # accuracy = [(sample_pred[i] == sample_test[i]).mean() for i in range(T)]
    # y_pred = np.array(y_probs) > 0.5
    # print("LogLoss = %.4f" % log_loss(y_test, y_pred))

    # # calculate mean log loss
    # logloss = [log_loss(sample_test[i], sample_pred[i]) for i in range(T)]
    # # compare distibution of output values

    # print("Test ROC-AUC: %.4f, test PR-AUC: %.4f" % (roc_auc_score(y_test, y_probs), 
    #                                                      average_precision_score(y_test, y_probs)))

    # # calculate mean AUC-ROC & AUC-PR
    # auc_roc = [roc_auc_score(sample_test[i], sample_prob[i]) for i in range(T)]
    # auc_pr = [average_precision_score(sample_test[i], sample_prob[i]) for i in range(T)]

    # mean_logloss, std_logloss = bootstrap_estimate_mean_stddev(logloss)
    # mean_accuracy, std_accuracy = bootstrap_estimate_mean_stddev(accuracy)
    # mean_auc_roc, std_auc_roc = bootstrap_estimate_mean_stddev(auc_roc)
    # mean_auc_pr, std_auc_pr = bootstrap_estimate_mean_stddev(auc_pr)
    
    # print("LogLoss:  mean = %.4f, std = %.4f" % (mean_logloss, std_logloss))
    # print("Accuracy: mean = %.4f, std = %.4f" % (mean_accuracy, std_accuracy))
    # print("AUC-ROC:  mean = %.4f, std = %.4f" % (mean_auc_roc, std_auc_roc))
    # print("AUC-PR:   mean = %.4f, std = %.4f" % (mean_auc_pr, std_auc_pr))   
