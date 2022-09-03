from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '../src')
from lib import datasets
from torch.utils.data import DataLoader

import torch
from glob import glob
import os.path as op
import os
from os.path import join as opj
import pandas as pd

import nibabel as nib
import importlib
import numpy as np

from nilearn import plotting, image, datasets
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def image_to_tensor(img):
    '''
    Convert Nifti1Image or load filename and return tensor to be input to the model.

    Parameters:
        - img, str or Nifti1Image

    Return:
        - sample, FloatTensor
    '''

    if isinstance(img, nib.Nifti1Image):
        sample = img
    else:
        sample = nib.load(img)

    sample = sample.get_data().copy().astype(float)
    sample = np.nan_to_num(sample)

    sample = torch.tensor(sample).view((1), (1), *sample.shape)
    
    return sample.type('torch.FloatTensor')

def recreate_image(img_mat, affine, header, out_img = False):
    '''
    Recreate image based on an matrix, affine and header. 
    '''
    tmp_data = img_mat.detach().numpy().reshape(*img_mat.shape[2:])
    
    if out_img:
        norm_img_data = tmp_data.copy().astype(float)
        norm_img_data = np.nan_to_num(norm_img_data)
        norm_img_data *= 1.0/np.abs(norm_img_data).max()
        img_data = norm_img_data
        
    else:
        img_data = tmp_data

    img = nib.Nifti1Image(img_data, affine=affine, header=header)

    return img

def plot_image_comparison(in_img, out_img, idx):
    '''
    Plot comparison of original and reconstructed image.
    '''
    f = plt.figure(figsize = (15, 7))
    gs = f.add_gridspec(2, 1)
    ax1 = f.add_subplot(gs[0, 0])
    ax2 = f.add_subplot(gs[1, 0])
    f.suptitle(f"Reconstruction of image {idx} with AutoEncoder",
               fontweight = 'bold')
    plotting.plot_stat_map(in_img, cut_coords = [-13, 1, 19], title = f"Original", figure = f, axes = ax1)
    plotting.plot_stat_map(out_img, cut_coords = [-13, 1, 19], title = f"Reconstruction", figure = f, axes = ax2)
    #plt.show()
    
    return f

def mask_using_original(inim, outim):
    '''
    Compute the mask of the original map and apply it to the reconstructed one. 
    '''
    # Set masking using NaN's
    data_orig = inim.get_fdata()
    data_repro = outim.get_fdata()
    
    if np.any(np.isnan(data_orig)):
        data_nan_orig = data_orig
        data_nan_repro = data_repro
        
        data_nan_repro[np.isnan(data_orig)] = np.nan
    else:
        data_nan_orig = data_orig
        data_nan_repro = data_repro

        data_nan_repro[data_orig == 0] = np.nan
        data_nan_orig[data_orig == 0] = np.nan
        
    # Save as image
    data_img_nan_orig = nib.Nifti1Image(data_nan_orig, inim.affine)
    data_img_nan_repro = nib.Nifti1Image(data_nan_repro, outim.affine)

    return data_img_nan_orig, data_img_nan_repro

def get_correlation(inim, outim):
    '''
    Compute the Pearson's correlation coefficient between original and reconstructed images.
    '''
    orig, repro = mask_using_original(inim, outim)
    
    data1 = orig.get_fdata().copy()
    data2 = repro.get_fdata().copy()
    
    # Vectorise input data
    data1 = np.reshape(data1, -1)
    data2 = np.reshape(data2, -1)

    in_mask_indices = np.logical_not(
        np.logical_or(
            np.logical_or(np.isnan(data1), np.absolute(data1) == 0),
            np.logical_or(np.isnan(data2), np.absolute(data2) == 0)))

    data1 = data1[in_mask_indices]
    data2 = data2[in_mask_indices]
    
    corr_coeff = np.corrcoef(data1, data2)[0][1]
    
    return corr_coeff

def tester(test_dataset, model_parameter, plot=False):
    '''
    Testing loop to apply on all test dataset.
    '''
    model_parameter = model_parameter.eval()

    correlation_list = []

    for idx, img in enumerate(sorted(test_dataset.data)):
        tmp = nib.load(img)
        affine = tmp.affine.copy()
        header = tmp.header

        in_tensor = image_to_tensor(tmp)
        out_tensor = model_parameter(in_tensor)

        in_img = recreate_image(in_tensor, affine, header)
        out_img = recreate_image(out_tensor, affine, header, out_img = True)
        
        correlation = get_correlation(in_img, out_img)

        correlation_list.append(correlation) 

        if plot:
            plot_image_comparison(in_img, out_img, idx)

    correlation_df = pd.Series(correlation_list)
    
    return correlation_df