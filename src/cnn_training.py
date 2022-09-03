# To test
from lib import datasets, cnn_trainer
from torch.utils.data import DataLoader
import torch.utils.data as data 
import torch
from glob import glob
import os.path as op
import os
from os.path import join as opj
import sys
import getopt
import json
import pickle
import importlib
import numpy as np
import warnings

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=RuntimeWarning)

if __name__ == "__main__":
    data_dir = None
    out_dir = None
    preprocess_type = None
    subset = None
    epochs = None
    model_to_use = None
    batch_size = None
    lr = None
    retrain = None
    classif = None


    try:
        OPTIONS, REMAINDER = getopt.getopt(sys.argv[1:], 'o:d:e:b:p:s:m:l:r:c:', ['out_dir=', 'data_dir=', 'epochs=', 'batch_size=', 
            'preprocess_type=', 'subset=', 'model=', 'learning_rate=', 'retrain=', 'classif='])

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    # Replace variables depending on options
    for opt, arg in OPTIONS:
        if opt in ('-o', '--out_dir'):
            out_dir= arg
        elif opt in ('-d', '--data_dir'):
            data_dir = arg
        elif opt in ('-p', '--preprocess_type'): 
            preprocess_type = str(arg)
        elif opt in ('-s', '--subset'): 
            subset = str(arg)
        elif opt in ('-m', '--model'): 
            model_to_use = str(arg)
        elif opt in ('-e', '--epochs'):
            epochs = int(arg)
        elif opt in ('-b', '--batch_size'):
            batch_size = int(arg)
        elif opt in ('-l', '--learning_rate'):
            lr = float(arg)
        elif opt in ('-r', '--retrain'):
            retrain = str(arg)
        elif opt in ('-c', '--classif'):
            classif = str(arg)

    print('OPTIONS   :', OPTIONS)

    assert(preprocess_type in ['resampled', 'resampled_masked', 'resampled_normalized', 'resampled_masked_normalized'])
    assert(type(batch_size)==int)
    assert(type(epochs)==int)
    assert(type(lr)==float)   
    
    if data_dir and out_dir and preprocess_type and subset and epochs and batch_size and model_to_use and lr and classif:
        str_lr = "{:.0e}".format(lr)
        package = 'lib.' + model_to_use
        md = importlib.import_module(package)

        train_id_file = opj(data_dir, f'train_{subset}.txt')
        valid_id_file = opj(data_dir, f'valid_{subset}.txt')
        label_file = opj(data_dir, f'{subset}_labels.csv')

        label_column = classif

        train_set = datasets.ClassifDataset(opj(data_dir, preprocess_type), 
                                          train_id_file, label_file, label_column)

        valid_set = datasets.ClassifDataset(opj(data_dir, preprocess_type), 
                                          valid_id_file, label_file, label_column)

        if retrain == 'all':
            pretrained_dict = opj(out_dir, f'neurovault_dataset_maps_{preprocess_type}_global_set_epochs_200_batch_size_32_{model_to_use}_lr_1e-04',
                'model_final.pt')

            cnn_trainer.finetuning(pretrained_dict, model_to_use, train_set, valid_set,
                        opj(out_dir, f"{subset}_maps_classification_{classif}_{model_to_use}_retrain_{retrain}_{preprocess_type}_epochs_{epochs}_batch_size_{batch_size}_lr_{str_lr}"), 
                        epochs, batch_size, lr)

        elif retrain == 'all_kfold':
            pretrained_dict = opj(out_dir, f'neurovault_dataset_maps_{preprocess_type}_global_set_epochs_200_batch_size_32_{model_to_use}_lr_1e-04',
                'model_final.pt')
            cnn_trainer.finetuning_with_cv(pretrained_dict, model_to_use, train_set, valid_set, opj(out_dir, 
                        f"{subset}_maps_classification_{classif}_{model_to_use}_retrain_{retrain}_validation_{preprocess_type}_epochs_{epochs}_batch_size_{batch_size}_lr_{str_lr}"),
                        epochs, batch_size, lr)

        elif retrain == 'kfold':
            cnn_trainer.training_with_cv(model_to_use, train_set, valid_set, opj(out_dir, 
                        f"{subset}_maps_classification_{classif}_{model_to_use}_retrain_no_validation_{retrain}_{preprocess_type}_epochs_{epochs}_batch_size_{batch_size}_lr_{str_lr}"),
                        epochs, batch_size, lr)

        else: # normal training
            cnn_trainer.training(model_to_use, train_set, valid_set, opj(out_dir, 
                        f"{subset}_maps_classification_{classif}_{model_to_use}_retrain_{retrain}_{preprocess_type}_epochs_{epochs}_batch_size_{batch_size}_lr_{str_lr}"),
                        epochs, batch_size, lr)




            

		


		
