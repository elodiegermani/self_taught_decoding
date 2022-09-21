import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import os.path as op
from os.path import join as opj
import sys
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from lib import model_cnn_4layers, model_cnn_5layers, datasets
from sklearn import metrics
import importlib
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import KFold


def train(model, train_dataset, distance, optimizer, device):
    '''
    Function to perform training of the CNN. 

    Parameters:
        - model, Classifier3D object: model trained.
        - train_dataset, ImageDataset object: training dataset.
        - distance: loss function used during training.
        - optimizer: optimizer used during training
        - device, torch.device: which device is beeing used.

    Returns:
        - mean_loss, float: mean loss on training dataset.
        - acc, float: accuracy on training dataset.
    '''
    model.train()
    total_preds = []
    total_labels = []
    with torch.set_grad_enabled(True):
        loss_total = 0

        for idx, data in enumerate(train_dataset):
            x_train = data[0]
            x_train = x_train.float().to(device)
            y_train = data[1]
            y_train = y_train.float().to(device)

            # clearing the Gradients of the model parameters
            optimizer.zero_grad()
            
            # prediction for training and validation set
            output_train = model(x_train)

            preds = torch.max(output_train, dim=1)[1]
            labels = torch.max(y_train, 1)[1]
            
            for i in preds:
                total_preds.append(i.item())
            for j in labels:
                total_labels.append(j.item())

            # computing the training and validation loss
            loss_train = distance(output_train, torch.max(y_train, 1)[1])
            loss_total += loss_train.item()

            # computing the updated weights of all the model parameters
            loss_train.backward()
            optimizer.step()

            if not (idx % 100):
                print(idx, loss_train.data, flush=True)

        acc = metrics.accuracy_score(y_true = total_labels, y_pred = total_preds)
        print('Training accuracy:', acc)

        mean_loss = loss_total / len(train_dataset)

    return mean_loss, acc

def validate(model, valid_dataset, distance, device):
    '''
    Function to perform validation during training of the autoencoder. 

    Parameters:
        - model, Classifier object: model trained.
        - valid_dataset, ImageDataset object: validation dataset.
        - distance: loss function used during training.
        - device, torch.device: which device is beeing used.

    Returns:
        - mean_loss, float: mean loss on validation dataset.
        - acc, float: accuracy on validation dataset
    '''
    model.eval()

    loss_total = 0
    total_preds = []
    total_labels = []

    for idx, data in enumerate(valid_dataset):
        x_val = data[0]
        x_val = x_val.float().to(device)
        y_val = data[1]
        y_val = y_val.float().to(device)
        
        # prediction for training and validation set
        output_val = model(x_val)

        preds = torch.max(output_val, dim=1)[1]
        labels = torch.max(y_val, 1)[1]
        
        for i in preds:
            total_preds.append(i.item())
        for j in labels:
            total_labels.append(j.item())

        # computing the training and validation loss
        loss_val = distance(output_val, torch.max(y_val, 1)[1])
        loss_total += loss_val.item()

    mean_loss = loss_total / len(valid_dataset)

    acc = metrics.accuracy_score(y_true = total_labels, y_pred = total_preds)
    print('Accuracy:', acc)

    return mean_loss, acc

def training(model_to_use, data_dir, subset, valid, label_column, preprocess_type, out_dir, epochs, batch_size, lr=1e-4):
    '''
    Function to perform training FROM SCRATCH of the Classifier. 

    Parameters:
        - model_to_use, string: model to train
        - train_subset, ClassifDataset: training dataset
        - valid_subset, ClassifDataset: validation dataset
        - out_dir, str: directory where to store the results
        - epochs, int: how many epochs to train
        - batch_size, int: size of batch to use
        - lr, float: learning rate of the optimizer
    '''
    # Reproducibility constraints
    package = 'lib.' + model_to_use
    md = importlib.import_module(package)

    random_seed=0
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
    #Create Tensorboard writer for logging
    if not os.path.isdir(out_dir): os.mkdir(out_dir)

    distance = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(random_seed)
        print('Using GPU.')
    else:
        device = "cpu"
        print('Using CPU.')

    for i in range(5):

        print(f'Fold {i}')

        train_id_file = opj(data_dir, f'train_{valid}_{subset}_fold_{i}.txt')
        valid_id_file = opj(data_dir, f'test_{valid}_{subset}_fold_{i}.txt')

        label_file = opj(data_dir, f'{subset}_labels.csv')
        label_list = sorted(np.unique(pd.read_csv(label_file)[label_column]))

        train_subset = datasets.ClassifDataset(opj(data_dir, preprocess_type), 
                                          train_id_file, label_file, label_column, label_list)

        valid_subset = datasets.ClassifDataset(opj(data_dir, preprocess_type), 
                                          valid_id_file, label_file, label_column, label_list)

        print(f'Train: {len(train_subset.data)}')
        print(f'Test: {len(valid_subset.data)}')

        n_class = len(train_subset.label_list)

        model = md.Classifier3D(n_class)

        print('Model:', model)
        model = model.to(device)
        print(f'Optimizer: ADAM, lr={lr}')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
        
        outfname = op.join(out_dir, 'model')

        train_dataset = DataLoader(train_subset, batch_size=batch_size)
        valid_dataset = DataLoader(valid_subset, batch_size=batch_size)

        print('Start training...')
        training_loss = []
        validation_loss = []

        for epoch in range(epochs):
            current_training_loss, acc = train(model, train_dataset, distance, optimizer, device)
            training_loss.append(current_training_loss)

            current_validation_loss, acc = validate(model, valid_dataset, distance, device)
            validation_loss.append(current_validation_loss)

            print('Training loss:', current_training_loss)
            print('Validation loss:', current_validation_loss)
            print('Validation accuracy:', acc)

            print('')
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, current_training_loss))

            if device != 'cpu':
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        print('Final accuracy:', acc)
        torch.save(model,  f"{outfname}_fold_{i}.pt")

    print('Training ended')

def finetuning(pretrained_dict, model_to_use, data_dir, subset, valid, label_column, preprocess_type, out_dir, epochs, batch_size, lr=1e-4):
    '''
    Function to perform training WITH FINETUNING of the Classifier. 

    Parameters:
        - pretrained_dict, str: filepath to the file containing the state dictionnary of the pre-trained model
        - model_to_use, string: model to train
        - train_subset, ClassifDataset: training dataset
        - valid_subset, ClassifDataset: validation dataset
        - out_dir, str: directory where to store the results
        - epochs, int: how many epochs to train
        - batch_size, int: size of batch to use
        - lr, float: learning rate of the optimizer
    '''
    # Reproducibility constraints
    package = 'lib.' + model_to_use
    md = importlib.import_module(package)

    random_seed=0

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
    #Create Tensorboard writer for logging
    if not os.path.isdir(out_dir): os.mkdir(out_dir)

    outfname = op.join(out_dir, 'model')

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(random_seed)
    else:
        device = "cpu"
        
    for i in range(5):

        print(f'Fold {i}')

        train_id_file = opj(data_dir, f'train_{valid}_{subset}_fold_{i}.txt')
        valid_id_file = opj(data_dir, f'test_{valid}_{subset}_fold_{i}.txt')

        label_file = opj(data_dir, f'{subset}_labels.csv')
        label_list = sorted(np.unique(pd.read_csv(label_file)[label_column]))

        train_subset = datasets.ClassifDataset(opj(data_dir, preprocess_type), 
                                          train_id_file, label_file, label_column, label_list)

        valid_subset = datasets.ClassifDataset(opj(data_dir, preprocess_type), 
                                          valid_id_file, label_file, label_column, label_list)

        print(f'Train: {len(train_subset.data)}')
        print(f'Test: {len(valid_subset.data)}')

        train_dataset = DataLoader(train_subset, batch_size=batch_size)
        valid_dataset = DataLoader(valid_subset, batch_size=batch_size)

        pretrained_model = torch.load(pretrained_dict, map_location=device)
        pretrained_model = pretrained_model.state_dict()

        # Initialize the model with the pre-trained weights
        model = md.Encoder3D()
        
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_model = {k: v for k, v in pretrained_model.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_model) 
        # 3. load the new state dict
        model.load_state_dict(pretrained_model)

        n_class = len(np.unique(train_subset.label_list))


        # Add layers corresponding to the correct architecture and classification
        if isinstance(model, model_cnn_5layers.Encoder3D):
            out_feature = 4096
            model.deconv5 = nn.Sequential(
                    nn.Flatten(start_dim=1),
                    nn.Linear(out_feature, n_class), 
                    nn.Softmax())
        else:
            out_feature = 512*3*4*3
            model.deconv4 = nn.Sequential(
                    nn.Flatten(start_dim=1),
                    nn.Linear(out_feature, n_class), 
                    nn.Softmax())
        
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
            
        distance = nn.CrossEntropyLoss()
            
        print('Start training...')

        for epoch in range(epochs):
            current_training_loss, acc = train(model, train_dataset, distance, optimizer, device)

            current_validation_loss,acc = validate(model, valid_dataset, distance, device)

            print('Training loss:', current_training_loss)
            print('Validation loss:', current_validation_loss)
            print('Validation accuracy:', acc)

            print('')
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1,
                                                      epochs, current_training_loss))

            if device != 'cpu':
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        print('Final accuracy:', acc)
        torch.save(model,  f"{outfname}_fold_{i}.pt")   

    print('Training ended')
