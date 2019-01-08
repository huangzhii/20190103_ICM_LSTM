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
import utils, LSTM, json
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc, logging, copy, pickle, math, random, argparse, time
import matplotlib.pyplot as plt
import optunity

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs to train for. Default: 300")
    parser.add_argument('--optunity_iteration', default=100, type=int)
    parser.add_argument('--box_optunity', default="/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/hyperparams_box_constraints.json")
    parser.add_argument('--gap', default=6, type=int)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--results_dir', default='/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/Results/LSTM_20190107_normalized', help="results dir")
    return parser.parse_args()

def load_box_constraints(file):
    with open(file, 'rb') as fp:
        return json.loads(fp.read())
    
def training(X_train, y_train, X_test, y_test, lr, l1_reg, epochs, weight_decay, hidden_size, num_layers, batch_size, dropout, device):
    # =============================================================================
    #     Model
    # =============================================================================
    model = LSTM.LSTM(input_size = X_train.shape[2], hidden_size = hidden_size, \
                      num_layers = num_layers, batch_size = batch_size, num_classes = 2, \
                      dropout = dropout, device = device)
    #model.hidden = model.init_hidden()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # define l2 penalty below, not at here.
    
    dataloader = DataLoader(torch.FloatTensor(X_train), batch_size=batch_size, pin_memory=True, shuffle=False)
    lblloader = DataLoader(torch.LongTensor(y_train), batch_size=batch_size, pin_memory=True, shuffle=False)
    
    auc_train_list, auc_test_list, f1_train_list, f1_test_list, \
        precision_test_list, recall_test_list = [], [], [], [], [], []
    for idx in tqdm(range(epochs)):
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
        
    return auc_train_list, auc_test_list, f1_train_list, f1_test_list, \
            precision_test_list, recall_test_list, outputs_test_bin, \
            outputs_test_proba, loss_test, fpr, tpr, model


def objective_function(log_learning_rate, hidden_size, nof_hidden_layer, l1_reg=0, dropout_rate=0):
    global optunity_iteration, datasets, num_epochs, batch_size, weight_decay, device, logger
    optunity_iteration += 1
    print("Optunity iteration: ", optunity_iteration)
    logger.log(logging.INFO, "Optunity iteration: %d" % optunity_iteration)
    
    lr = 10**log_learning_rate
    num_layers = int(nof_hidden_layer) # floor
    hidden_size = int(hidden_size) # floor
    
    print("lr: %.4E; n_layers: %d; n_hidden: %d; l1_reg: %.2E; dropout: %.2f" % \
          (lr, num_layers, hidden_size, l1_reg, dropout_rate))
    logger.log(logging.INFO, "lr: %.4E; n_layers: %d; n_hidden: %d; l1_reg: %.2E; dropout: %.2f" % \
          (lr, num_layers, hidden_size, l1_reg, dropout_rate))
    
    X_train = datasets['train']['X']
    y_train = datasets['train']['y']
    X_test = datasets['val']['X']
    y_test = datasets['val']['y']
    
    auc_train_list, auc_test_list, f1_train_list, f1_test_list, \
    precision_test_list, recall_test_list, outputs_test_bin, \
    outputs_test_proba, loss_test, fpr, tpr, model = \
            training(X_train, y_train, X_test, y_test, lr, l1_reg, num_epochs, \
                     weight_decay, hidden_size, num_layers, batch_size, dropout_rate, device)
            
    print("MIMIC AUC: %.8f; Macro F1: %.8f; Precision: %.8f; Recall: %.8f" % \
          (auc_test_list[-1], f1_test_list[-1], precision_test_list[-1], recall_test_list[-1]))
    logger.log(logging.INFO, "MIMIC AUC: %.8f; Macro F1: %.8f; Precision: %.8f; Recall: %.8f" % \
               (auc_test_list[-1], f1_test_list[-1], precision_test_list[-1], recall_test_list[-1]))
    
    return f1_test_list[-1]



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
    plt.ioff()
    
    args.results_dir = args.results_dir + "/LSTM_optunity" + str(args.optunity_iteration) + \
        "_ep=" + str(args.num_epochs) + "_l2=" + str(args.l2)

    #device = torch.device('cpu')
    fname = 'Data_5folds_6_%d_1_20190107.pickle' % args.gap
    print("Running with dataset Gap = %d" % args.gap)
    datasets_5folds = pickle.load( open( '/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/data_5folds_no_CA_normalized_not_consider_uniqueID/' + fname, "rb" ) )
    # Load and normalize EICU data
    data_EICU = pd.read_csv("/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/EICU_final_data_for_LSTM_20190102.csv")
    X_EICU, y_EICU, column_names_EICU, icustay_id_EICU = \
        utils.preprocessing(data_EICU, series = 6, gap = args.gap)
    scaler = pickle.load(open('/home/zhihuan/Documents/20181207_Hypoxemia/20190103_ICM_LSTM/data/scaler.pickle', 'rb') )
    data_EICU.loc[:, column_names_EICU] = scaler.transform(data_EICU.loc[:, column_names_EICU])
    X_EICU, y_EICU, column_names_EICU, icustay_id_EICU = \
        utils.preprocessing(data_EICU, series = 6, gap = args.gap)
        
    # Feature Ranking
    mypath = args.results_dir + '/Gap_' + str(args.gap)
    if not os.path.exists(mypath + '/feature_ranking'):
        os.makedirs(mypath + '/feature_ranking')
    auc_test_feature_ranking_list = pd.DataFrame(np.zeros((X_EICU.shape[2]+1, len(datasets_5folds)*2)))
    f1_test_feature_ranking_list = pd.DataFrame(np.zeros((X_EICU.shape[2]+1, len(datasets_5folds)*2)))
    precision_test_feature_ranking_list = pd.DataFrame(np.zeros((X_EICU.shape[2]+1, len(datasets_5folds)*2)))
    recall_test_feature_ranking_list = pd.DataFrame(np.zeros((X_EICU.shape[2]+1, len(datasets_5folds)*2)))
    
    
    for i in range(len(datasets_5folds)):
        print("%d fold CV -- %d/%d" % (len(datasets_5folds), i+1, len(datasets_5folds)))
        TIMESTRING  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())
        results_dir_dataset = args.results_dir + '/Gap_' + str(args.gap) + \
                                '/run_' + TIMESTRING + '_fold_' + str(i+1)
        if not os.path.exists(results_dir_dataset):
            os.makedirs(results_dir_dataset)
        
            
        box_constraints = load_box_constraints(args.box_optunity)
        global optunity_iteration, datasets, num_epochs, batch_size, weight_decay, device, logger
        optunity_iteration = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Device configuration
        weight_decay = args.l2
        num_epochs = args.num_epochs
        batch_size = 2**20
        datasets = datasets_5folds[str(i+1)]
        
        # create logger
        logger = logging.getLogger(TIMESTRING)
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(results_dir_dataset+'/mainlog.log', mode='w')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.log(logging.INFO, "Arguments: %s" % args)
        
# =============================================================================
#         Optunity start
# =============================================================================
        
        print('Box Constraints: ' + str(box_constraints))
        logger.log(logging.INFO, 'Box Constraints: %s' % str(box_constraints))
        print('Maximizing F1 score. Num_iterations: %d' % args.optunity_iteration)
        logger.log(logging.INFO, 'Maximizing F1 score. Num_iterations: %d' % args.optunity_iteration)
        
        opt_params, call_log, _ = optunity.maximize(objective_function, num_evals=args.optunity_iteration,
                                                    solver_name='sobol', **box_constraints)
 
        lr = 10**opt_params['log_learning_rate']
        hidden_size = int(opt_params['hidden_size'])
        nof_hidden_layer = int(opt_params['nof_hidden_layer'])
        dropout_rate = opt_params['dropout_rate']
        l1_reg = opt_params['l1_reg']
        
        
        print("[Optimal Parameters] lr: %.4E; n_layers: %d; n_hidden: %d; l1_reg: %.4E; dropout: %.4f" % \
              (lr, nof_hidden_layer, hidden_size, l1_reg, dropout_rate))
        logger.log(logging.INFO, "[Optimal Parameters] lr: %.4E; n_layers: %d; n_hidden: %d; l1_reg: %.4E; dropout: %.4f" % \
              (lr, nof_hidden_layer, hidden_size, l1_reg, dropout_rate))
        
# =============================================================================
#         Final Training Start
# =============================================================================
        
        X_train = np.concatenate((datasets['train']['X'], datasets['val']['X']), 0)
        y_train = np.concatenate((datasets['train']['y'], datasets['val']['y']), 0)
        X_test = datasets['test']['X']
        y_test = datasets['test']['y']
        
        auc_train_list, auc_test_list, f1_train_list, f1_test_list, \
        precision_test_list, recall_test_list, outputs_test_bin, \
        outputs_test_proba, loss_test, fpr, tpr, model = \
                training(X_train, y_train, X_test, y_test, lr, l1_reg, num_epochs, \
                         weight_decay, hidden_size, nof_hidden_layer, batch_size, dropout_rate, device)
            
        print("MIMIC AUC: %.8f; Macro F1: %.8f; Precision: %.8f; Recall: %.8f" % \
              (auc_test_list[-1], f1_test_list[-1], precision_test_list[-1], recall_test_list[-1]))
        logger.log(logging.INFO, "MIMIC AUC: %.8f; Macro F1: %.8f; Precision: %.8f; Recall: %.8f" % \
                   (auc_test_list[-1], f1_test_list[-1], precision_test_list[-1], recall_test_list[-1]))
        
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
        plt.plot(range(num_epochs), auc_train_list, "r-",linewidth=1)
        plt.plot(range(num_epochs), auc_test_list, "g--",linewidth=1)
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
        
# =============================================================================
#   Feature Ranking after 5 fold CV
# =============================================================================
    
        for j in tqdm(range(X_test.shape[2]+1)):
#            print(j)
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
            
            auc_test_feature_ranking_list.loc[j, i] = auc_test
            f1_test_feature_ranking_list.loc[j, i] = f1_test
            precision_test_feature_ranking_list.loc[j, i] = precision
            recall_test_feature_ranking_list.loc[j, i] = recall
            
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
            
            auc_test_feature_ranking_list.loc[j, len(datasets_5folds)+i] = auc_EICU
            f1_test_feature_ranking_list.loc[j, len(datasets_5folds)+i] = f1
            precision_test_feature_ranking_list.loc[j, len(datasets_5folds)+i] = precision
            recall_test_feature_ranking_list.loc[j, len(datasets_5folds)+i] = recall
            
    auc_test_feature_ranking_list.columns = np.asarray(["MIMIC_fold1", "MIMIC_fold2", "MIMIC_fold3", "MIMIC_fold4", "MIMIC_fold5", \
                                        "EICU_fold1", "EICU_fold2", "EICU_fold3", "EICU_fold4", "EICU_fold5"])
    auc_test_feature_ranking_list.index = np.concatenate((['All features'], column_names_EICU))
    f1_test_feature_ranking_list.columns, precision_test_feature_ranking_list.columns, recall_test_feature_ranking_list.columns = \
        auc_test_feature_ranking_list.columns, auc_test_feature_ranking_list.columns, auc_test_feature_ranking_list.columns
    f1_test_feature_ranking_list.index, precision_test_feature_ranking_list.index, recall_test_feature_ranking_list.index = \
        auc_test_feature_ranking_list.index, auc_test_feature_ranking_list.index, auc_test_feature_ranking_list.index
    
    auc_test_feature_ranking_list.to_csv(mypath + '/feature_ranking/AUC.csv')
    f1_test_feature_ranking_list.to_csv(mypath + '/feature_ranking/F1.csv')
    precision_test_feature_ranking_list.to_csv(mypath + '/feature_ranking/Precision.csv')
    recall_test_feature_ranking_list.to_csv(mypath + '/feature_ranking/Recall.csv')