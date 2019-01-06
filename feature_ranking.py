#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:21:06 2018

@author: zhihuan
"""

import sys, os
sys.path.append("/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM")
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import pandas as pd
import utils, LSTM
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc, logging, copy, pickle, math, random, argparse, time
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs to train for. Default: 300")
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--gap', default=6, type=int)
    parser.add_argument('--results_dir', default='/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/Results/LSTM_20190104', help="results dir")
    return parser.parse_args()

if __name__=='__main__':
    gc.collect()
    torch.cuda.manual_seed_all(666)
    torch.manual_seed(666)
    random.seed(666)
    np.random.seed(666)
    torch.cuda.empty_cache()
    args = parse_args()
    criterion = nn.CrossEntropyLoss()
    softmax = F.softmax
    epochs = args.num_epochs
    plt.ioff()
#    run_fold = args.run_fold
    
    batch_size = 2**20
    learning_rate = 1e-2
    weight_decay = 0
    args.results_dir = args.results_dir + "/LSTM_" + str(args.hidden_size) + "_" + str(args.num_layers)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    mypath = args.results_dir + '/Gap_' + str(args.gap)
    if not os.path.exists(mypath + '/feature_ranking'):
        os.makedirs(mypath + '/feature_ranking')
    run_folds_dirs = os.listdir(mypath)
    run_folds_dirs = [s for s in run_folds_dirs if 'run_' in s]
    
    print("Running with dataset Gap = %d" % args.gap)
    fname = 'Data_5folds_6_%d_1_20190103.pickle' % args.gap
    datasets_5folds = pickle.load( open( '/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/' + fname, "rb" ) )
    data_EICU = pd.read_csv("/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/EICU_final_data_for_LSTM_20190102.csv")
    X_EICU, y_EICU, column_names_EICU, icustay_id_EICU = \
        utils.preprocessing(data_EICU, series = 6, gap = args.gap)
    
    auc_test_list = pd.DataFrame(np.zeros((datasets_5folds['1']['train']['X'].shape[2]+1, len(run_folds_dirs)*2)))
    f1_test_list = pd.DataFrame(np.zeros((datasets_5folds['1']['train']['X'].shape[2]+1, len(run_folds_dirs)*2)))
    precision_test_list = pd.DataFrame(np.zeros((datasets_5folds['1']['train']['X'].shape[2]+1, len(run_folds_dirs)*2)))
    recall_test_list = pd.DataFrame(np.zeros((datasets_5folds['1']['train']['X'].shape[2]+1, len(run_folds_dirs)*2)))

    
    for i, foldpath in enumerate(run_folds_dirs):
        print("%d fold CV -- %d/%d" % (len(datasets_5folds), i+1, len(datasets_5folds)))
        # =============================================================================
        #     Model
        # =============================================================================
        model = LSTM.LSTM(input_size = 43, hidden_size = args.hidden_size, num_layers = args.num_layers, batch_size = batch_size, num_classes = 2, device = device)
        model.load_state_dict(pickle.load(open( mypath + '/' + foldpath + '/model.pickle', 'rb') ))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # define l2 penalty below, not at here.

        datasets = datasets_5folds[str(i+1)]
        X_train = np.concatenate((datasets['train']['X'], datasets['val']['X']), 0)
        y_train = np.concatenate((datasets['train']['y'], datasets['val']['y']), 0)
        X_test = datasets['test']['X']
        y_test = datasets['test']['y']
        
        for j in range(X_test.shape[2]+1):
            print(j)
            X_test_new = copy.deepcopy(X_test)
            X_EICU_new = copy.deepcopy(X_EICU)
            if j >= 1: # the first one is the original data
                X_test_new[:,:,j-1] = 0
                X_EICU_new[:,:,j-1] = 0
        
                
            outputs_test = model(torch.FloatTensor(X_test_new).to(device))
            outputs_test_proba = softmax(outputs_test).cpu().data.numpy()[:,1]
            outputs_test_bin = np.argmax(softmax(outputs_test).cpu().data.numpy(),1)
            fpr, tpr, thresholds = roc_curve(y_test, outputs_test_proba)
            auc_test = auc(fpr, tpr)
            f1_test = f1_score(outputs_test_bin, y_test, average = 'macro')
            precision = precision_score(y_test, outputs_test_bin, average = 'macro')
            recall = recall_score(y_test, outputs_test_bin, average = 'macro')
            
            auc_test_list.loc[j, i] = auc_test
            f1_test_list.loc[j, i] = f1_test
            precision_test_list.loc[j, i] = precision
            recall_test_list.loc[j, i] = recall
            
            # =============================================================================
            # Validate EICU
            # =============================================================================
            
            outputs_EICU_test = model(torch.FloatTensor(X_EICU_new).to(device))
            outputs_EICU_test_proba = softmax(outputs_EICU_test).cpu().data.numpy()[:,1]
            outputs_EICU_test_bin = np.argmax(softmax(outputs_EICU_test).cpu().data.numpy(),1)
            
            fpr, tpr, thresholds = roc_curve(y_EICU, outputs_EICU_test_proba)
            auc_EICU = auc(fpr, tpr)
            f1 = f1_score(outputs_EICU_test_bin, y_EICU, average = 'macro')
            precision = precision_score(y_EICU, outputs_EICU_test_bin, average = 'macro')
            recall = recall_score(y_EICU, outputs_EICU_test_bin, average = 'macro')
            
            auc_test_list.loc[j, len(run_folds_dirs)+i] = auc_EICU
            f1_test_list.loc[j, len(run_folds_dirs)+i] = f1
            precision_test_list.loc[j, len(run_folds_dirs)+i] = precision
            recall_test_list.loc[j, len(run_folds_dirs)+i] = recall
    
    
    auc_test_list.columns = np.asarray(["MIMIC_fold1", "MIMIC_fold2", "MIMIC_fold3", "MIMIC_fold4", "MIMIC_fold5", \
                                        "EICU_fold1", "EICU_fold2", "EICU_fold3", "EICU_fold4", "EICU_fold5"])
    auc_test_list.index = np.concatenate((['All features'], column_names_EICU))
    f1_test_list.columns, precision_test_list.columns, recall_test_list.columns = auc_test_list.columns, auc_test_list.columns, auc_test_list.columns
    f1_test_list.index, precision_test_list.index, recall_test_list.index = auc_test_list.index, auc_test_list.index, auc_test_list.index
    
    auc_test_list.to_csv(mypath + '/feature_ranking/AUC.csv')
    f1_test_list.to_csv(mypath + '/feature_ranking/F1.csv')
    precision_test_list.to_csv(mypath + '/feature_ranking/Precision.csv')
    recall_test_list.to_csv(mypath + '/feature_ranking/Recall.csv')
        