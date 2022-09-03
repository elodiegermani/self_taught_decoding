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


def validate(model, valid_dataset, distance, device):
    '''
    Function to perform validation during training of the autoencoder. 

    Parameters:
        - model, AutoEncoder3D object: model trained.
        - valid_dataset, ImageDataset object: validation dataset.
        - distance: loss function used during training.
        - device, torch.device: which device is beeing used.

    Returns:
        - mean_loss, float: mean loss on validation dataset.
    '''
    # Settings
    model.eval()
    loss_total = 0

    # Test validation data
    with torch.no_grad():
        for idx, data in enumerate(valid_dataset):
            img = data
            img = img.float().to(device)
            output = model(img)
            loss = distance(output, img)
            loss_total += loss.item()

    mean_loss = loss_total / len(valid_dataset)

    return mean_loss

def train(model, train_dataset, distance, optimizer, device):
    '''
    Function to perform training of the autoencoder.

    Parameters:
        - model, AutoEncoder3D object: model trained.
        - train_dataset, ImageDataset object: training dataset.
        - distance: loss function used during training.
        - optimizer: optimizer used during training
        - device, torch.device: which device is beeing used.

    Returns:
        - mean_loss, float: mean loss on training dataset.
    '''
    model.train()
    loss_total = 0

    # Train 
    for idx, data in enumerate(train_dataset):
        img = data
        img = img.float().to(device)
        # forward
        output = model(img)
        loss = distance(output, img)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not (idx % 100):
            print(idx, loss.data, flush=True)
        loss_total += loss.item()

    mean_loss = loss_total / len(train_dataset)

    return mean_loss 

# Perform training
def trainer(model, train_subset, valid_subset, out_dir, epochs, batch_size, lr=1e-4):
    '''
    Function to perform training of AutoEncoder. 

    Parameters:
        - model, AutoEncoder3D: model to train
        - train_subset, ImageDataset: training dataset
        - valid_subset, ImageDataset: validation dataset
        - out_dir, str: directory where to store the results
        - epochs, int: how many epochs to train
        - batch_size, int: size of batch to use
        - lr, float: learning rate of the optimizer
    '''
    # Reproducibility constraints
    random_seed=2
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
    #Create Tensorboard writer for logging
    if not os.path.isdir(out_dir): os.mkdir(out_dir)
    tb_dir = opj(out_dir, 'logs')
    if not os.path.isdir(tb_dir): os.mkdir(tb_dir)
    tb_writer = SummaryWriter(tb_dir)

    # Load data
    train_dataset = DataLoader(train_subset, batch_size=batch_size)
    valid_dataset = DataLoader(valid_subset, batch_size=batch_size)
    
    training_loss = []
    validation_loss = []

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Using GPU.')
    else:
        device = "cpu"
        print('Using CPU.')

    # Setup
    model = model.to(device)
    distance = nn.MSELoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
    
    outfname = op.join(out_dir, 'model')

    print('Start training...')
    for epoch in range(epochs):
        # Training loop
        current_training_loss = train(model, train_dataset, distance, optimizer, device)
        training_loss.append(current_training_loss)

        # Validation loop
        current_validation_loss = validate(model, valid_dataset, distance, device)
        validation_loss.append(current_validation_loss)

        print('Training loss:', current_training_loss)
        print('Validation loss:', current_validation_loss)

        tb_writer.add_scalars('Loss', {'train':training_loss[epoch],'validation':validation_loss[epoch]}, epoch)

        print('')
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, current_training_loss))

        if device != 'cpu':
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    torch.save(model,  "{0}_final.pt".format(outfname))

    tb_writer.close()




