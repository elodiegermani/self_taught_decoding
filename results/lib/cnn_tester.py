from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '../src')
from lib import datasets
from sklearn import metrics
from os.path import join as opj

import numpy as np
import pandas as pd

import os
import torch
import sys

def f1_score(outputs, labels):
    labs = sorted(np.unique(labels))
    
    f1 = metrics.f1_score(y_true = labels, y_pred= outputs, labels = labs, average = 'macro')
    
    return f1

def accuracy(outputs, labels):

    acc = metrics.accuracy_score(y_true = labels, y_pred = outputs)
    return acc

def precision(outputs, labels):
    labs = sorted(np.unique(labels))
    
    precision = metrics.precision_score(y_true = labels, y_pred= outputs, average = 'macro', labels = labs)
    
    return precision

def recall(outputs, labels):
    labs = sorted(np.unique(labels))
    
    rec = metrics.recall_score(y_true = labels, y_pred= outputs, average = 'macro', labels = labs)
    
    return rec

def accuracy_per_class(outputs, labels):
    labs = sorted(np.unique(labels))

    df = pd.DataFrame()
    true_label = []
    good_pred = []

    for label in labs: 
        n_true = len([l for l in labels if l == label])
        true_label.append(n_true)
        n_good_pred = 0
        for i, el in enumerate(labels):
            if outputs[i] == el and el==label:
                n_good_pred += 1 
        good_pred.append(n_good_pred)

    df['Labels'] = labs 
    df['True labels'] = true_label
    df['Good predictions'] = good_pred

    return df
                
def tester(test_dataset, parameter_file, device='cpu'):
    test_set = DataLoader(test_dataset, batch_size = 32, shuffle=False)
    model_parameter = torch.load(parameter_file, map_location=device)
        
    model_parameter = model_parameter.eval()
    
    total_preds = []
    total_labels = []
    
    for idx, data in enumerate(test_set):
        y_test = data[1].float()
        x_test = data[0].float()
        pred_test = model_parameter(x_test)
        
        preds = torch.max(pred_test, dim=1)[1]
        labels = torch.max(y_test, 1)[1]
        
        for i in preds:
            total_preds.append(i.item())
        for j in labels:
            total_labels.append(j.item())
            
                
    acc = accuracy(total_preds, total_labels)
    f1 = f1_score(total_preds, total_labels)
    prec = precision(total_preds, total_labels)
    rec = recall(total_preds, total_labels)
    acc_class = accuracy_per_class(total_preds, total_labels)
    
    return acc, f1, prec, rec, acc_class