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
    parser.add_argument('--num_epochs', type=int, default=400, help="Number of epochs to train for. Default: 300")
    parser.add_argument('--hidden_size', default=8, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--gap', default=6, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--l1', default=1e-4, type=float)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--results_dir', default='/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/Results/LSTM_20190105', help="results dir")
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
    
    learning_rate = args.lr
    weight_decay = args.l2
    l1_reg = args.l1
    dropout = args.dropout
    batch_size = 2**20
    
    args.results_dir = args.results_dir + "/LSTM_" + str(args.hidden_size) + "_" + \
        str(args.num_layers) + "_ep=" + str(epochs) + "_dr=" + str(dropout) + \
        "_lr=" + str(learning_rate) + "_l2=" + str(weight_decay) + \
        "_l1=" + str(l1_reg)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    fname = 'Data_5folds_6_%d_1_20190105_no_CA.pickle' % args.gap
    print("Running with dataset Gap = %d" % args.gap)
    datasets_5folds = pickle.load( open( '/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/' + fname, "rb" ) )
    data_EICU = pd.read_csv("/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/EICU_final_data_for_LSTM_20190102.csv")

    
    for i in range(len(datasets_5folds)):
        print("%d fold CV -- %d/%d" % (len(datasets_5folds), i+1, len(datasets_5folds)))
#        if run_fold != i+1:
#            continue
        # dataset
        TIMESTRING  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())
        
        results_dir_dataset = args.results_dir + '/Gap_' + str(args.gap) + \
                                '/run_' + TIMESTRING + '_fold_' + str(i+1)
        if not os.path.exists(results_dir_dataset):
            os.makedirs(results_dir_dataset)
            
        # =============================================================================
        #     Model
        # =============================================================================
        model = LSTM.LSTM(input_size = datasets_5folds['1']['test']['X'].shape[2], hidden_size = args.hidden_size, \
                          num_layers = args.num_layers, batch_size = batch_size, num_classes = 2, \
                          dropout = dropout, device = device)
        #model.hidden = model.init_hidden()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # define l2 penalty below, not at here.
        
        text_file = open(args.results_dir + "/parameter_setting.txt", "w")
        text_file.write(str(model) + "\n")
        text_file.write(str(optimizer) + "\n")
        text_file.write("Gap: %d\n" % args.gap)
        text_file.write("Batch size: %d\n" % batch_size)
        text_file.write("Number of Epochs: %d\n" % epochs)
        text_file.write("Dropout: %.4f\n" % dropout)
        text_file.write("L1 reg: %.4f\n" % l1_reg)
        text_file.write("L2 reg: %.4f\n" % weight_decay)
        text_file.close()
                
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
        dataloader = DataLoader(torch.FloatTensor(X_train), batch_size=batch_size, pin_memory=True, shuffle=False)
        lblloader = DataLoader(torch.LongTensor(y_train), batch_size=batch_size, pin_memory=True, shuffle=False)
        
        
        auc_train_list, auc_test_list, f1_train_list, f1_test_list, \
            precision_test_list, recall_test_list = [], [], [], [], [], []
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
                f1_train = f1_score(outputs_bin, lbl.cpu().data.numpy(), average = 'macro')
                f1_train_list.append(f1_train)
                loss = criterion(outputs, lbl)
#                L1 penalization
                l1_crit = nn.L1Loss(size_average=False)
                l1_loss = 0
                for param in model.parameters():
                    target = torch.FloatTensor(np.zeros(param.shape)).to(device)
                    l1_loss += l1_crit(param, target)
                loss += l1_reg * l1_loss
                
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
            precision = precision_score(y_test, outputs_test_bin, average = 'macro')
            precision_test_list.append(precision)
            recall = recall_score(y_test, outputs_test_bin, average = 'macro')
            recall_test_list.append(recall)
            
            print("iter: %03d, train loss: %.8f, test loss: %.8f" % (idx, loss.cpu().data.numpy(), loss_test.cpu().data.numpy()))
            print("         , train AUC: %.8f, test AUC: %.8f" % (auc_train, auc_test))
            print("         , train F-1: %.8f, test F-1: %.8f" % (f1_train, f1_test))
            logger.log(logging.INFO, "iter: %03d, train loss: %.8f, test loss: %.8f" % (idx, loss.cpu().data.numpy(), loss_test.cpu().data.numpy()))
            logger.log(logging.INFO, "         , train AUC: %.8f, test AUC: %.8f" % (auc_train, auc_test))
            logger.log(logging.INFO, "         , train F-1: %.8f, test F-1: %.8f" % (f1_train, f1_test))
        
        
        
        with open(results_dir_dataset + '/model.pickle', 'wb') as handle:
            pickle.dump(model.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + '/model.pickle', 'wb') as handle:
            pickle.dump(model.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + '/outputs_test_proba.pickle', 'wb') as handle:
            pickle.dump(outputs_test_proba, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + '/outputs_test_bin.pickle', 'wb') as handle:
            pickle.dump(outputs_test_bin, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        res_table = pd.DataFrame(list(zip(auc_train_list, auc_test_list, f1_train_list, f1_test_list, precision_test_list, recall_test_list)))
        res_table.columns = ['auc_train', 'auc_test', 'f1_train', 'f1_test', 'precision_test', 'recall_test']
        res_table.to_csv(results_dir_dataset + '/MIMIC_AFPR_table.csv')
        
        plt.figure(figsize=(8,4))
        plt.plot(range(epochs), auc_train_list, "r-",linewidth=1)
        plt.plot(range(epochs), auc_test_list, "g--",linewidth=1)
        plt.legend(['train', 'test'])
        plt.xlabel("epochs")
        plt.ylabel("AUC")
        plt.title("AUC Curve")
        plt.savefig(results_dir_dataset + "/MIMIC_convergence.png",dpi=300)
        
        plt.figure(figsize=(8,4))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='LSTM test set (AUC = %0.4f%%)' % (100*auc(fpr, tpr)))
        
        plt.axes().set_aspect('equal')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
#        plt.show()
        plt.savefig(results_dir_dataset + "/MIMIC_test_AUC.png",dpi=300)
    
        # =============================================================================
        # Validate EICU
        # =============================================================================
        
        X_EICU, y_EICU, column_names_EICU, icustay_id_EICU = \
            utils.preprocessing(data_EICU, series = 6, gap = args.gap)
#        unique, counts = np.unique(y_EICU, return_counts=True)
        outputs_EICU_test = model(torch.FloatTensor(X_EICU).to(device))
        outputs_EICU_test_proba = softmax(outputs_EICU_test).cpu().data.numpy()[:,1]
        outputs_EICU_test_bin = np.argmax(softmax(outputs_EICU_test).cpu().data.numpy(),1)
        
        with open(results_dir_dataset + '/outputs_EICU_test_proba.pickle', 'wb') as handle:
            pickle.dump(outputs_EICU_test_proba, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + '/outputs_EICU_test_bin.pickle', 'wb') as handle:
            pickle.dump(outputs_EICU_test_bin, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        fpr, tpr, thresholds = roc_curve(y_EICU, outputs_EICU_test_proba)
        auc_EICU = auc(fpr, tpr)
        f1 = f1_score(outputs_EICU_test_bin, y_EICU, average = 'macro')
        precision = precision_score(y_EICU, outputs_EICU_test_bin, average = 'macro')
        recall = recall_score(y_EICU, outputs_EICU_test_bin, average = 'macro')
        
        print("EICU AUC: %.8f; Macro F1: %.8f; Precision: %.8f; Recall: %.8f" % (auc_EICU, f1, precision, recall))
        logger.log(logging.INFO, "EICU AUC: %.8f; Macro F1: %.8f; Precision: %.8f; Recall: %.8f" % (auc_EICU, f1, precision, recall))
        
        plt.figure(figsize=(8,8))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='LSTM EICU set (AUC = %0.4f%%)' % (100*auc(fpr, tpr)))
        plt.axes().set_aspect('equal')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic Curve (EICU)')
        plt.legend(loc="lower right")
#        plt.show()
        plt.savefig(results_dir_dataset + "/EICU_AUC.png",dpi=300)
        