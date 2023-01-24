import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
import matplotlib.pyplot as plt 
import numpy as np
from nilearn import datasets, plotting
import nibabel as nib


def get_correlation(data1, data2):
	'''
	Compute the Pearson's correlation coefficient between original and reconstructed images.
	'''
	
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

def visualize_features(parameters_file, dataset, classe, classe_name, types, subjects = [], print_title=True):

	model=torch.load(parameters_file, 'cpu')
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
	#print(f"Total convolution layers: {counter}")
	#print(conv_layers)

	all_img = [[] for i in range(5)]
	all_img_full = [[] for i in range(5)]

	image_loader = DataLoader(dataset, batch_size = 1, shuffle=False)
	for i, data in enumerate(image_loader):
		if dataset.get_original_labels()[i] == classe:
			image = data[0].float()

			outputs = []
			names = []
			for layer in conv_layers[0:]:
				image = layer(image)
				outputs.append(image)
				names.append(str(layer))
			#print(len(outputs))
			#print feature_maps
			#for feature_map in outputs:
				#print(feature_map.shape)

			processed = []
			for feature_map in outputs:
				feature_map = feature_map.squeeze(0)
				gray_scale = torch.sum(feature_map,0)
				gray_scale = gray_scale / feature_map.shape[0]
				processed.append(gray_scale.data.cpu().numpy())
			#for fm in processed:
				#print(fm.shape)

			all_img[0].append(data[0].cpu().numpy()[0,0,:,:,int(data[0].cpu().numpy().shape[2]/2)])
			all_img_full[0].append(data[0].cpu().numpy()[0,0,:,:,:])

			for i in range(1, len(all_img)):
				all_img[i].append(processed[i-1][:,:,int(processed[i-1].shape[2]/2)])
				all_img_full[i].append(processed[i-1][:,:,:])

	correlations = [[] for i in range(5)]
	for c in range(len(correlations)):
		for i, img1 in enumerate(all_img_full[c]):
			for j, img2 in enumerate(all_img_full[c]):  
				if i != j:
					correlations[c].append(get_correlation(img1, img2))

	fig = plt.figure(figsize=(25, 5))
	if print_title:
		fig.suptitle(f'{classe_name}', fontsize=28, weight='bold')
	for i in range(len(all_img)):
		a = fig.add_subplot(1,5, i+1)
		mean_img = np.mean(all_img[i], axis=0)

		imgplot = plt.imshow(mean_img, cmap = nilearn_cmaps['cold_hot'], )
		#plt.colorbar()
		a.axis("off")
		if print_title:
			if i == 0:
				a.set_title(f'ORIGINAL IMAGE', fontsize=20, weight='bold')
			else:
				a.set_title(f'LAYER {i}', fontsize=20, weight='bold')
	plt.savefig(f'../figures/mean_features_{classe}_{types}.png')
	plt.show()

	if len(subjects) > 0:
		sub_list = [dataset.get_original_subject()[s] for s, c in enumerate(dataset.get_original_labels()) if c==classe]
		for subs in subjects:
			idx = sub_list.index(subs)
			features = [all_img[i][idx] for i in range(len(all_img))]
			fig = plt.figure(figsize=(25, 4))
			fig.suptitle(f'{classe}, {subs}', fontsize=24, weight='bold')
			for l in range(len(features)):
				a = fig.add_subplot(1,5, l+1)
				imgplot = plt.imshow(features[l], cmap = nilearn_cmaps['cold_hot'])
				plt.colorbar()
				a.axis("off")
				if l == 0:
					a.set_title(f'Original image', fontsize=20,  weight='bold')
				else:
					a.set_title(f'Layer {l}', fontsize=20, weight='bold')
			plt.savefig(f'../figures/subject_{subs}_features_{classe}_{types}.png')
			plt.show()

	return correlations