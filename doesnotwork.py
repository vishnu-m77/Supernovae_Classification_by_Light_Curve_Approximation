import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim




# Binary Classification 

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 8, 3)
        self.conv2 = nn.Conv1d(6, 8, 3, padding=1)
        self.layer = nn.LayerNorm((8, 128))
        self.fc1 = nn.ReLU()
        
        self.pool = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1)
        self.layer2 = nn.LayerNorm((16, 64))
        self.fc2 = nn.ReLU()
        
        self.pool2 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1)
        self.layer3 = nn.LayerNorm((32, 32))
        self.fc4 = nn.ReLU()
        
        self.pool3 = nn.MaxPool1d(2)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.33)
        self.fc5 = nn.Linear(16 * 32, 1)
        self.sig = nn.Sigmoid()
                                

    def forward(self, x):
        x = (self.fc1(self.layer(self.conv2(self.conv1(x)))))
        x = (self.fc2(self.layer2(self.conv3(self.pool(x)))))
        x = (self.fc4(self.layer3(self.conv4(self.pool2(x)))))
        x = (self.sig(self.fc5(self.drop(self.flat(self.pool3(x))))))
        
        return x
        
