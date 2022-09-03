from torch.utils.data import Dataset
from glob import glob
from os.path import join as opj
from sklearn.model_selection import train_test_split
from torch import default_generator, randperm, Generator
from torch._utils import _accumulate
import nibabel as nib
import pandas as pd
import numpy as np
import torch
import json
import random
import os 
import shutil

class ImageDataset(Dataset):
    '''
    Create a Dataset object used to load training data and train the autoencoder (no labels needed).

    Parameters:
        - data_dir, str: directory where images are stored
        - id_file, str: path to the text file containing ids of images of interest

    Attributes:
        - data, list of str: list containing all images of the dataset selected
        - ids, list of int: list containing all ids of images of the selected dataset
    '''
    def __init__(self, data_dir, id_file):
        id_list = []
        with open(id_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                id_list.append(line[:-1])
        file.close()
        
        id_list = [str(int(i)) for i in id_list]

        global_file_list = glob(opj(data_dir, '*.nii*'))
        file_list = sorted([f for f in global_file_list if str(int(os.path.basename(f).split('.')[0])) in id_list])

        N = len(file_list)
        assert(N == len(id_list))

        self.data = file_list
        self.ids = id_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.data[idx]
        sample = nib.load(fname).get_fdata().copy().astype(float)
        sample = np.nan_to_num(sample)

        sample = torch.tensor(sample).view((1), *sample.shape)
        
        return sample

    def get_original_ids(self):
        return self.ids

class ClassifDataset(Dataset):
    '''
    Create a Dataset object used to load training data and train a model using pytorch.

    Parameters:
        - data_dir, str: directory where images are stored
        - id_file, str: path to the text file containing ids of images of interest
        - label_file, str: path to the csv file containing labels of images of interest
        - label_column, str: name of the column to use as labels in label_file
        - label_list, list: list of unique labels sorted in alphabetical order

    Attributes:
        - data, list of str: list containing all images of the dataset selected
        - ids, list of int: list containing all ids of images of the selected dataset
        - labels, list of str: list containing all labels of each data
    '''
    def __init__(self, data_dir, id_file, label_file, label_column, label_list):
        id_list = []
        with open(id_file, "r") as file:
                lines = file.readlines()
                for line in lines:
                    id_list.append(line[:-1])
        file.close()
        
        id_list = [str(int(i)) for i in id_list]

        global_file_list = glob(opj(data_dir, '*.nii*'))
        file_list = sorted([f for f in global_file_list if str(int(os.path.basename(f).split('.')[0])) in id_list])

        N = len(file_list)
        assert(N == len(id_list))

        self.data = file_list
        self.ids = sorted(id_list)
        self.labels = []
        self.classe = label_column
        self.subject = []

        label_df = pd.read_csv(label_file)

        for idx in self.ids:
            self.labels.append(label_df[self.classe][label_df['image_id'] == int(idx)].to_string(index=False))
            self.subject.append(label_df['subject'][label_df['image_id'] == int(idx)].to_string(index=False))

        self.label_list = sorted(label_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.data[idx]
        label = self.labels[idx]
        label_vect = [0 for i in range(len(self.label_list))]

        for i in range(len(self.label_list)):
            if label == self.label_list[i]:
                label_vect[i] = 1
        sample = nib.load(fname).get_fdata().copy().astype(float)
        sample = np.nan_to_num(sample)

        sample = torch.tensor(sample).view((1), *sample.shape)
        label_vect = torch.tensor(label_vect)
        
        return sample, label_vect

    def get_original_ids(self):
        return self.ids

    def get_original_labels(self):
        return self.labels

    def get_original_subject(self):
        return self.subject

class ConcatenateDatasets(Dataset):
    def __init__(self, dataset1, dataset2):
        assert(dataset1.label_list == dataset2.label_list)
        
        self.data = dataset1.data + dataset2.data
        self.ids = dataset1.ids + dataset2.ids 
        self.labels = dataset1.labels + dataset2.labels
        self.subject = dataset1.subject + dataset2.subject 
        self.label_list = sorted(dataset1.label_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.data[idx]
        label = self.labels[idx]
        label_vect = [0 for i in range(len(self.label_list))]

        for i in range(len(self.label_list)):
            if label == self.label_list[i]:
                label_vect[i] = 1
        sample = nib.load(fname).get_fdata().copy().astype(float)
        sample = np.nan_to_num(sample)

        sample = torch.tensor(sample).view((1), *sample.shape)
        label_vect = torch.tensor(label_vect)
        
        return sample, label_vect

    def get_original_ids(self):
        return self.ids

    def get_original_labels(self):
        return self.labels

    def get_original_subject(self):
        return self.subject


class Dataset_Fold(Dataset):
    def __init__(self, data, ids, labels, subject, label_list):
        self.data = data
        self.ids = ids 
        self.labels = labels
        self.subject = subject 
        self.label_list = sorted(label_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.data[idx]
        label = self.labels[idx]
        label_vect = [0 for i in range(len(self.label_list))]

        for i in range(len(self.label_list)):
            if label == self.label_list[i]:
                label_vect[i] = 1
        sample = nib.load(fname).get_fdata().copy().astype(float)
        sample = np.nan_to_num(sample)

        sample = torch.tensor(sample).view((1), *sample.shape)
        label_vect = torch.tensor(label_vect)
        
        return sample, label_vect

    def get_original_ids(self):
        return self.ids

    def get_original_labels(self):
        return self.labels

    def get_original_subject(self):
        return self.subject

class KFoldDatasets(Dataset):
    def __init__(self, dataset, n_val):
        self.subject = dataset.get_original_subject()
        self.sub_list = sorted(np.unique(dataset.get_original_subject()).tolist())
        self.sub_val=[self.sub_list[i:i + n_val] for i in range(0, len(self.sub_list), n_val)]
        self.valid_folds = []
        self.train_folds = []

        label_list = dataset.label_list

        for i, val_set in enumerate(self.sub_val):
            val_data = []
            val_ids = []
            val_labels = []
            val_subject = []

            for s, sub in enumerate(self.subject):
                if sub in val_set:
                    val_data.append(dataset.data[s])
                    val_ids.append(dataset.ids[s])
                    val_labels.append(dataset.labels[s])
                    val_subject.append(dataset.subject[s])

            train_data = [d for d in dataset.data if d not in val_data]
            train_ids = [d for d in dataset.ids if d not in val_ids]
            train_labels = [dataset.labels[d] for d in range(len(dataset.labels)) if dataset.ids[d] not in val_ids]
            train_subject = [d for d in dataset.subject if d not in val_subject]

            self.valid_folds.append(Dataset_Fold(val_data, val_ids, val_labels, val_subject, label_list))
            self.train_folds.append(Dataset_Fold(train_data, train_ids, train_labels, train_subject, label_list))




