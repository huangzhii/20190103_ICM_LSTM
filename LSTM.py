import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import pandas as pd
import utils
import LSTM
from sklearn.metrics import auc, roc_curve, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
import copy
#code based on https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py

# Define the model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, num_classes, dropout, device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device # GPU or CPU?
#        self.hidden = self.init_hidden()
#        
#    def init_hidden(self):
#        # Set initial hidden and cell states 
#        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
#        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
#        h0 = torch.FloatTensor(h0).to(device)
#        c0 = torch.FloatTensor(c0).to(device)
#        return (h0,c0)
        
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
#        out, self.hidden = self.lstm(x, self.hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode hidden state of last time step
        out2 = self.fc(out[:, -1, :])
        return out2