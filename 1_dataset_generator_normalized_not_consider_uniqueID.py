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
from sklearn import preprocessing


random.seed(666)
np.random.seed(666)


gc.collect()
gap = 6

data = pd.read_csv("/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/MIMIC_final_data_for_LSTM_20190102.csv")
data['GENDER'][data['GENDER'] == 'F'] = 0
data['GENDER'][data['GENDER'] == 'M'] = 1

normalization = True

X, y, column_names, icustay_id = utils.preprocessing(data, series = 6, gap = gap)
uniqueID = np.unique(icustay_id)

if normalization:
    toscale = data.loc[:, column_names]
    scaler = preprocessing.StandardScaler().fit(toscale)
    data.loc[:, column_names] = scaler.transform(toscale)
    
    X, y, column_names, icustay_id = utils.preprocessing(data, series = 6, gap = gap)
    uniqueID = np.unique(icustay_id)
    
    with open('/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/data_5folds_no_CA_normalized_not_consider_uniqueID/scaler.pickle', 'wb') as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)



kf = KFold(n_splits=5, random_state=666, shuffle=True)
i = 0
datasets_folds = {}

for train_index, test_index in kf.split(y):
    i += 1
    idx = np.random.randint(len(train_index), size=round(len(train_index)/4))
    val_index = train_index[idx]
    train_index = np.delete(train_index, idx)
    
#    uniqueID_train = uniqueID[train_index]
#    uniqueID_val = uniqueID[val_index]
#    uniqueID_test = uniqueID[test_index]
#    index_train = [i for i, val in enumerate(icustay_id) if val in uniqueID_train]
#    index_val = [i for i, val in enumerate(icustay_id) if val in uniqueID_val]
#    index_test = [i for i, val in enumerate(icustay_id) if val in uniqueID_test]
    
    datasets = {}
    datasets['train'] = {}
    datasets['val'] = {}
    datasets['test'] = {}
    datasets['column_names'] = column_names
    
    datasets['train']['X'] = X[train_index, :, :].astype(np.double)
    datasets['train']['y'] = y[train_index].astype(np.int32)
    
    datasets['val']['X'] = X[val_index, :, :].astype(np.double)
    datasets['val']['y'] = y[val_index].astype(np.int32)
    
    datasets['test']['X'] = X[test_index, :, :].astype(np.double)
    datasets['test']['y'] = y[test_index].astype(np.int32)
    
    datasets_folds[str(i)] = datasets
processed_dir = '/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/data_5folds_no_CA_normalized_not_consider_uniqueID'

#### git ignore all pickle data, since they are too large!
with open(processed_dir + '/Data_5folds_6_%d_1_20190107.pickle' % gap, 'wb') as handle:
    pickle.dump(datasets_folds, handle, protocol=pickle.HIGHEST_PROTOCOL)