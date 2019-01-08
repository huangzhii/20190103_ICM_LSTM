#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:23:47 2019

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
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

torch.cuda.manual_seed_all(666)
torch.manual_seed(666)
random.seed(666)
np.random.seed(666)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gap', default=6, type=int)
    parser.add_argument('--results_dir', default='/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/Results/Traditional_20190105_normalized', help="results dir")
    return parser.parse_args()

if __name__=='__main__':
    gc.collect()
    random.seed(666)
    np.random.seed(666)
    args = parse_args()
    plt.ioff()
    
    fname = 'Data_5folds_6_%d_1_20190105_no_CA_normalized.pickle' % args.gap
    print("Running with dataset Gap = %d" % args.gap)
    datasets_5folds = pickle.load( open( '/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/' + fname, "rb" ) )
    # Load and normalize EICU data
    data_EICU = pd.read_csv("/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/EICU_final_data_for_LSTM_20190102.csv")
    X_EICU, y_EICU, column_names_EICU, icustay_id_EICU = \
        utils.preprocessing(data_EICU, series = 6, gap = args.gap)
    scaler = pickle.load(open('/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/scaler.pickle', 'rb') )
    data_EICU.loc[:, column_names_EICU] = scaler.transform(data_EICU.loc[:, column_names_EICU])
    X_EICU, y_EICU, column_names_EICU, icustay_id_EICU = \
        utils.preprocessing(data_EICU, series = 6, gap = args.gap)
        
        
    mtds = ["logit_l1", "logit_l2", "NN_l2", "AdaBoost", "GBC", "RFC"]
    mtds = ["logit_l2", "NN"]
    
    for mtd in mtds:
        if mtd == "logit_l1": # around 8 mins for all folds
            regr = LogisticRegression(penalty='l1', C=1) # 3.55
        if mtd == "logit_l2": # around 4 mins for all folds
            regr = LogisticRegression(penalty='l2', C=4) # 4.2
        if mtd == "NN": # around 2 mins for all folds
            regr = MLPClassifier(solver='adam', alpha=0, max_iter=2000, # alpha is L2 reg
                                 hidden_layer_sizes=(64), random_state=1)
        if mtd == "NN_l2": # around 2 mins for all folds
            regr = MLPClassifier(solver='adam', alpha=1e-5, max_iter=2000, # alpha is L2 reg
                                 hidden_layer_sizes=(64), random_state=1)
        if mtd == "SVC": # Too slow! cost at least 3 hours for one fold!
            regr = SVC(kernel='linear', C=1, probability = True)
        if mtd == "AdaBoost":
            regr = AdaBoostClassifier(n_estimators=100)
        if mtd == "GBC":
            regr = GradientBoostingClassifier(n_estimators=100)
        if mtd == "RFC":
            regr = RandomForestClassifier(n_estimators=300)
            
            
            
            
        for i in range(len(datasets_5folds)):
            print("%d fold CV -- %d/%d" % (len(datasets_5folds), i+1, len(datasets_5folds)))
        #        if run_fold != i+1:
        #            continue
            # dataset
            TIMESTRING  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())
            
            results_dir_dataset = args.results_dir + '/' + mtd + '/Gap_' + str(args.gap) + \
                                    '/run_' + TIMESTRING + '_fold_' + str(i+1)
            if not os.path.exists(results_dir_dataset):
                os.makedirs(results_dir_dataset)
        #        if not os.path.exists(results_dir_dataset + '/find_hyperparameters'):
        #            os.makedirs(results_dir_dataset + '/find_hyperparameters')
                    
            # create logger
            logger = logging.getLogger(TIMESTRING)
            logger.setLevel(logging.DEBUG)
            # create file handler which logs even debug messages
            fh = logging.FileHandler(results_dir_dataset+'/mainlog.log', mode='w')
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
        
            logger.log(logging.INFO, "Arguments: %s" % args)
            datasets = datasets_5folds[str(i+1)]
            
            X_train = np.concatenate((datasets['train']['X'], datasets['val']['X']), 0)
            y_train = np.concatenate((datasets['train']['y'], datasets['val']['y']), 0)
            X_test = datasets['test']['X']
            y_test = datasets['test']['y']
            
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
            
            regr.fit(X_train_reshaped, y_train)
            
            
            # AUC F1 train
            y_pred = regr.predict(X_train_reshaped)
            y_pred_proba = regr.predict_proba(X_train_reshaped)[:,1]
            fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba)
            auc_train = auc(fpr, tpr)
            f1_train = f1_score(y_pred, y_train, average = 'macro')
            
            # AUC F1 test
            y_pred = regr.predict(X_test_reshaped)
            y_pred_proba = regr.predict_proba(X_test_reshaped)[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            auc_test = auc(fpr, tpr)
            f1_test = f1_score(y_pred, y_test, average = 'macro')
            
            # P, R test
            precision_test = precision_score(y_test, y_pred, average = 'macro')
            recall_test = recall_score(y_test, y_pred, average = 'macro')
            
            print("[MIMIC] train AUC: %.8f, test AUC: %.8f, train F1: %.8f, test F1: %.8f, test Precision: %.8f, test Recall: %.8f" \
                  % (auc_train, auc_test, f1_train, f1_test, precision_test, recall_test))
            logger.log(logging.INFO, "[MIMIC] train AUC: %.8f, test AUC: %.8f, train F1: %.8f, test F1: %.8f, test Precision: %.8f, test Recall: %.8f" \
                  % (auc_train, auc_test, f1_train, f1_test, precision_test, recall_test))
            
            
            with open(results_dir_dataset + '/outputs_test_proba.pickle', 'wb') as handle:
                pickle.dump(y_pred_proba, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(results_dir_dataset + '/outputs_test_bin.pickle', 'wb') as handle:
                pickle.dump(y_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            res_table = pd.DataFrame([auc_train, auc_test, f1_train, f1_test, precision_test, recall_test])
            res_table.index = ['auc_train', 'auc_test', 'f1_train', 'f1_test', 'precision_test', 'recall_test']
            res_table.to_csv(results_dir_dataset + '/MIMIC_AFPR_table.csv')
            
            plt.figure(figsize=(8,4))
            plt.plot(fpr, tpr, color='darkorange',
                     lw=2, label='%s test set (AUC = %0.4f%%)' % (mtd, 100*auc(fpr, tpr)))
            plt.axes().set_aspect('equal')
            plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('Receiver Operating Characteristic Curve')
            plt.legend(loc="lower right")
            plt.savefig(results_dir_dataset + "/MIMIC_test_AUC.png",dpi=300)
        
            # =============================================================================
            # Validate EICU
            # =============================================================================
            
        #        unique, counts = np.unique(y_EICU, return_counts=True)
            X_EICU_reshaped = X_EICU.reshape(X_EICU.shape[0], -1)
            
            outputs_EICU_test_bin = regr.predict(X_EICU_reshaped)
            outputs_EICU_test_proba = regr.predict_proba(X_EICU_reshaped)[:,1]
            
            with open(results_dir_dataset + '/outputs_EICU_test_proba.pickle', 'wb') as handle:
                pickle.dump(outputs_EICU_test_proba, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(results_dir_dataset + '/outputs_EICU_test_bin.pickle', 'wb') as handle:
                pickle.dump(outputs_EICU_test_bin, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            fpr, tpr, thresholds = roc_curve(y_EICU, outputs_EICU_test_proba)
            auc_EICU = auc(fpr, tpr)
            f1 = f1_score(outputs_EICU_test_bin, y_EICU, average = 'macro')
            precision = precision_score(y_EICU, outputs_EICU_test_bin, average = 'macro')
            recall = recall_score(y_EICU, outputs_EICU_test_bin, average = 'macro')
            
            print("[EICU] AUC: %.8f; Macro F1: %.8f; Precision: %.8f; Recall: %.8f" % (auc_EICU, f1, precision, recall))
            logger.log(logging.INFO, "[EICU] AUC: %.8f; Macro F1: %.8f; Precision: %.8f; Recall: %.8f" % (auc_EICU, f1, precision, recall))
            
            plt.figure(figsize=(8,8))
            plt.plot(fpr, tpr, color='darkorange',
                     lw=2, label='%s EICU set (AUC = %0.4f%%)' % (mtd, 100*auc(fpr, tpr)))
            plt.axes().set_aspect('equal')
            plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('Receiver Operating Characteristic Curve (EICU)')
            plt.legend(loc="lower right")
            plt.savefig(results_dir_dataset + "/EICU_AUC.png",dpi=300)
            