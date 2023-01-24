from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from glob import glob
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker
from itertools import chain
from nilearn import datasets
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img, binarize_img, resample_img
from nilearn.masking import intersect_masks, apply_mask
from os.path import join as opj
import pandas as pd
import matplotlib.colors as mcolors
import random
import torch.nn as nn 

def resample(img, resolution=2):
    template = load_mni152_template(resolution=resolution)
    res_img = resample_to_img(img, template)

    return res_img

def get_intersection_mask(images_list):
    nii_img_list = []
    for img in images_list:
        nii_img_list.append(binarize_img(img))

    mask = intersect_masks(nii_img_list)

    return mask

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

def plot_latent_space(dataset, encoder=None, latent_space_size=512*2*2*2, cluster=True, n_cluster=2):
    colors=[]
    data = []
    model = None

    if encoder:
        model_parameter = torch.load(encoder, map_location="cpu")
        model = model_parameter.eval()
        
        # we will save the conv layer weights in this list
        model_weights =[]
        #we will save the 49 conv layers in this list
        conv_layers = []
        # get all the model children as list
        model_children = list(model.children())
        #counter to keep count of the conv layers
        counter = 0
        #append all the conv layers and their respective wights to the list
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv3d:
                counter+=1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv3d:
                            counter+=1
                            model_weights.append(child.weight)
                            conv_layers.append(child)
                            
        print(f"Total convolution layers: {counter}")
        print(conv_layers)

        data = np.zeros((len(list(dataset.data)), latent_space_size))
        for i,img in enumerate(sorted(dataset.data)):
            tmp = nib.load(img)
            affine = tmp.affine.copy()
            header = tmp.header

            affine[:3,:3] = np.sign(affine[:3,:3]) * 4
            shape = (48, 56, 48)

            tmp = resample_img(tmp, target_affine=affine, target_shape=shape, interpolation='nearest')

            in_tensor = image_to_tensor(tmp)
            outputs=[]
            for layer in conv_layers[0:]:
                in_tensor = layer(in_tensor)
                outputs.append(in_tensor)
                
            out_tensor = outputs[-1]
            data[i]= np.reshape(out_tensor.detach().numpy(), -1)

    else:
        data = np.zeros((len(list(dataset.data)), (48*56*48)))
        for i,img in enumerate(sorted(dataset.data)):
            tmp = nib.load(img)
            affine = tmp.affine.copy()
            header = tmp.header

            affine[:3,:3] = np.sign(affine[:3,:3]) * 4
            shape = (48, 56, 48)

            tmp = resample_img(tmp, target_affine=affine, target_shape=shape, interpolation='nearest')

            in_tensor = image_to_tensor(tmp)
            data[i]= np.reshape(in_tensor.detach().numpy(), -1)

    pca = PCA(n_components=2)
    pca = pca.fit_transform(data)

    if cluster:
        model = cluster_pca(pca, dataset, n_cluster)
        return pca, model
        
    else: 
        plot_pca(pca, dataset)
        return pca

def find_n_clusters(pca):
    pca_dataframe = pd.DataFrame(pca)
    wcss=[]
    for k in range(1, len(pca)):
        
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)
        
        # Fit model to samples
        model.fit(pca_dataframe)
        wcss.append(model.inertia_)

    plt.figure(figsize=(10, 8))
    plt.plot(range(1,len(pca)), wcss, marker='o', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('K-means with PCA Clustering')
    plt.show()

def cluster_pca(pca, dataset, n_clusters):
    col_list = list(mcolors.CSS4_COLORS.keys())
    #random.shuffle(col_list)
    pca_dataframe = pd.DataFrame(pca[:, 0:2], columns=['Comp 1', 'Comp 2'])
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=n_clusters)
    
    # Fit model to samples
    model.fit(pca_dataframe)

    pca_dataframe['K-means clusters']=model.labels_

    x,y=pca[:, 0:2].T

    annotations = []

    for i, img in enumerate(dataset.data):
        annotations.append(dataset.get_original_subject()[i] + '_' + dataset.get_original_labels()[i])
    
    
    f = plt.figure(figsize=(16, 16))
    for c in np.unique(pca_dataframe['K-means clusters'].tolist()):
        plt.scatter(pca_dataframe['Comp 1'][pca_dataframe['K-means clusters']==c], 
            pca_dataframe['Comp 2'][pca_dataframe['K-means clusters']==c], 
            c= col_list[c],
            label = f'Cluster {c}')

    #for i, label in enumerate(annotations):
        #plt.annotate(annotations[i], (x[i], y[i]))
    plt.legend()

    centers = model.cluster_centers_

    for c in range(len(centers)):
        plt.scatter(centers[c][0], centers[c][1], s=200, c=col_list[c], marker='s')
    plt.title('Clustering of group statistic maps on PCA components.')

    plt.show()
    f.savefig(f'../figures/clustering_maps.png')

    return model


def plot_pca(pca, dataset):
    col_list = list(mcolors.CSS4_COLORS.keys())
    #random.shuffle(col_list)
    pca_dataframe = pd.DataFrame(pca[:, 0:2], columns=['Comp 1', 'Comp 2'])
    
    labels = sorted(np.unique(dataset.get_original_labels()))
    
    labels_to_int = []
    
    for lab in dataset.get_original_labels():
        labels_to_int.append(labels.index(lab))

    pca_dataframe['Labels']=labels_to_int

    x,y=pca[:, 0:2].T

    annotations = []

    for i, img in enumerate(dataset.data):
        annotations.append(dataset.get_original_subject()[i] + '_' + dataset.get_original_labels()[i])
    
    
    f = plt.figure(figsize=(16, 16))
    for c in np.unique(labels_to_int):
        plt.scatter(pca_dataframe['Comp 1'][pca_dataframe['Labels']==c], 
            pca_dataframe['Comp 2'][pca_dataframe['Labels']==c], 
            c= col_list[c],
            label = f'Label {c}')

    #for i, label in enumerate(annotations):
        #plt.annotate(annotations[i], (x[i], y[i]))
    plt.legend()

    plt.show()
    f.savefig(f'../figures/plot_pca_maps.png')


    
