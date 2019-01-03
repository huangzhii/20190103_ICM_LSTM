#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:21:06 2018

@author: zhihuan
"""

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
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


torch.cuda.manual_seed_all(666)
torch.manual_seed(666)
random.seed(666)
np.random.seed(666)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

gc.collect()
data = pd.read_csv("data/MIMIC_final_data_for_LSTM_20190102.csv")

X, y, column_names, icustay_id = utils.preprocessing(data, series = 6, gap = 6)
uniqueID = np.unique(icustay_id)
index = np.random.permutation(uniqueID.shape[0])
uniqueID = uniqueID[index]
uniqueID_train = uniqueID[0:round(len(uniqueID)*0.9)]
uniqueID_test = uniqueID[round(len(uniqueID)*0.9):]

index_train = [i for i, val in enumerate(icustay_id) if val in uniqueID_train]
index_test = [i for i, val in enumerate(icustay_id) if val in uniqueID_test]
X_train = X[index_train, :, :]
X_test = X[index_test, :, :]
y_train = y[index_train]
y_test = y[index_test]




#mtds = ["logit_l1", "logit_l2", "NN_l2"]
#X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
#X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
#for mtd in mtds:
#    if mtd == "logit_l1":
#        regr = LogisticRegression(penalty='l1', C=1) # 3.55
#    if mtd == "logit_l2":
#        regr = LogisticRegression(penalty='l2', C=1) # 4.2
#    if mtd == "NN_l2":
#        regr = MLPClassifier(solver='adam', alpha=1e-5, max_iter=2000, # alpha is L2 reg
#                             hidden_layer_sizes=(64), random_state=1)
#    
#    regr.fit(X_train_reshaped, y_train)
#    y_pred = regr.predict(X_test_reshaped)
#    y_pred_proba = regr.predict_proba(X_test_reshaped)[:,1]
#    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
#    print(auc(fpr, tpr))



batch_size = 2**20
dataloader = DataLoader(torch.FloatTensor(X_train), batch_size=batch_size, pin_memory=True, shuffle=False)
lblloader = DataLoader(torch.LongTensor(y_train), batch_size=batch_size, pin_memory=True, shuffle=False)



model = LSTM.LSTM(input_size = 43, hidden_size = 32, num_layers = 1, batch_size = batch_size, num_classes = 2, device = device)
#model.hidden = model.init_hidden()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0) # define l2 penalty below, not at here.



criterion = nn.CrossEntropyLoss()
softmax = F.softmax
epochs = 200
auc_train_list = []
auc_test_list = []
f1_train_list = []
f1_test_list = []
for idx in range(epochs):
    for data, lbl in zip(dataloader, lblloader):
        optimizer.zero_grad()
        data = data.to(device)
        lbl = lbl.to(device)
        outputs = model(data)
        outputs_proba = softmax(outputs).cpu().data.numpy()[:,1]
        outputs_bin = np.argmax(softmax(outputs).cpu().data.numpy(),1)
        fpr, tpr, thresholds = roc_curve(lbl.cpu().data.numpy(), outputs_proba)
        auc_train = auc(fpr, tpr)
        auc_train_list.append(auc_train)
        f1_train = f1_score(outputs_bin, lbl, average = 'macro')
        f1_train_list.append(f1_train)
        loss = criterion(outputs, lbl)
        
        loss.backward()
        optimizer.step()
        
    outputs_test = model(torch.FloatTensor(X_test).to(device))
    loss_test = criterion(outputs_test, torch.LongTensor(y_test).to(device))
    outputs_test_proba = softmax(outputs_test).cpu().data.numpy()[:,1]
    outputs_test_bin = np.argmax(softmax(outputs_test).cpu().data.numpy(),1)
    fpr, tpr, thresholds = roc_curve(y_test, outputs_test_proba)
    auc_test = auc(fpr, tpr)
    auc_test_list.append(auc_test)
    f1_test = f1_score(outputs_test_bin, y_test, average = 'macro')
    f1_test_list.append(f1_test)
    
    print("iter: %03d, train loss: %.4f, test loss: %.4f" % (idx, loss.cpu().data.numpy(), loss_test.cpu().data.numpy()))
    print("         , train AUC: %.4f, test AUC: %.4f" % (auc_train, auc_test))
    print("         , train F-1: %.4f, test F-1: %.4f" % (f1_train, f1_test))
    
    
    
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.plot(range(epochs), auc_train_list, "r-",linewidth=1)
plt.plot(range(epochs), auc_test_list, "g--",linewidth=1)
plt.legend(['train', 'test'])
plt.xlabel("epochs")
plt.ylabel("AUC")
title = "AUC Curve"
plt.title(title)
plt.savefig("convergence.png",dpi=300)

plt.figure(figsize=(8,4))
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='LSTM test set (AUC = %0.2f%%)' % (100*auc(fpr, tpr)))

plt.axes().set_aspect('equal')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()
plt.savefig("test_AUC.png",dpi=300)





# =============================================================================
# Validate EICU
# =============================================================================

data_EICU = pd.read_csv("data/EICU_final_data_for_LSTM_20190102.csv")
X_EICU, y_EICU, column_names_EICU, icustay_id_EICU = utils.preprocessing(data_EICU, series = 6, gap = 6)
unique, counts = np.unique(y_EICU, return_counts=True)

outputs_EICU_test = model(torch.FloatTensor(X_EICU).to(device))
outputs_EICU_test_proba = softmax(outputs_EICU_test).cpu().data.numpy()[:,1]
outputs_EICU_test_bin = np.argmax(softmax(outputs_EICU_test).cpu().data.numpy(),1)
fpr, tpr, thresholds = roc_curve(y_EICU, outputs_EICU_test_proba)
auc = auc(fpr, tpr)
f1 = f1_score(outputs_EICU_test_bin, y_EICU, average = 'macro')


#acc = sum(outputs_EICU_test_bin)/len(y_EICU)

print("AUC: %.4f; F1: %.4f" % (auc, f1))