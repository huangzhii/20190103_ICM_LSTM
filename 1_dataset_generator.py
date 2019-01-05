#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 11:05:03 2019

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
from sklearn.metrics import auc, roc_curve, f1_score
import gc
from sklearn.model_selection import KFold
import pickle


torch.cuda.manual_seed_all(666)
torch.manual_seed(666)
random.seed(666)
np.random.seed(666)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

gc.collect()
data = pd.read_csv("data/MIMIC_final_data_for_LSTM_20190102.csv")
gap = 1
X, y, column_names, icustay_id = utils.preprocessing(data, series = 6, gap = gap)
uniqueID = np.unique(icustay_id)

#index = np.random.permutation(uniqueID.shape[0])
#uniqueID = uniqueID[index]
#uniqueID_train = uniqueID[0:round(len(uniqueID)*0.9)]
#uniqueID_test = uniqueID[round(len(uniqueID)*0.9):]
#
#index_train = [i for i, val in enumerate(icustay_id) if val in uniqueID_train]
#index_test = [i for i, val in enumerate(icustay_id) if val in uniqueID_test]
#X_train = X[index_train, :, :]
#X_test = X[index_test, :, :]
#y_train = y[index_train]
#y_test = y[index_test]


kf = KFold(n_splits=5, random_state=666, shuffle=True)
i = 0
datasets_folds = {}

for train_index, test_index in kf.split(uniqueID):
    i += 1
    idx = np.random.randint(len(train_index), size=round(len(train_index)/4))
    val_index = train_index[idx]
    train_index = np.delete(train_index, idx)
    
    uniqueID_train = uniqueID[train_index]
    uniqueID_val = uniqueID[val_index]
    uniqueID_test = uniqueID[test_index]
    index_train = [i for i, val in enumerate(icustay_id) if val in uniqueID_train]
    index_val = [i for i, val in enumerate(icustay_id) if val in uniqueID_val]
    index_test = [i for i, val in enumerate(icustay_id) if val in uniqueID_test]
    
    datasets = {}
    datasets['train'] = {}
    datasets['val'] = {}
    datasets['test'] = {}
    datasets['column_names'] = column_names
    
    datasets['train']['X'] = X[index_train, :, :].astype(np.double)
    datasets['train']['y'] = y[index_train].astype(np.int32)
    datasets['train']['ICUSTAY_ID'] = uniqueID_train
    
    datasets['val']['X'] = X[index_val, :, :].astype(np.double)
    datasets['val']['y'] = y[index_val].astype(np.int32)
    datasets['val']['ICUSTAY_ID'] = uniqueID_val
    
    datasets['test']['X'] = X[index_test, :, :].astype(np.double)
    datasets['test']['y'] = y[index_test].astype(np.int32)
    datasets['test']['ICUSTAY_ID'] = uniqueID_test
    
    datasets_folds[str(i)] = datasets
processed_dir = '/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data'

#### git ignore all pickle data, since they are too large!
with open(processed_dir + '/Data_5folds_6_%d_1_20190103.pickle' % gap, 'wb') as handle:
    pickle.dump(datasets_folds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    

#### git ignore all pickle data, since they are too large!
with open(processed_dir + '/Data_5folds_6_%d_1_20190103.pickle' % gap, 'wb') as handle:
    pickle.dump(datasets_folds, handle, protocol=pickle.HIGHEST_PROTOCOL)