# On the benefits of self-taught learning for brain decoding

We study the benefits of using a large public neuroimaging database composed of fMRI statistic maps, in a self-taught learning framework, for improving brain decoding on new tasks. First, we leverage the NeuroVault database to train, on a selection of relevant statistic maps, a convolutional autoencoder to reconstruct these maps. Then, we use this trained encoder to initialize a supervised convolutional neural network to classify tasks or cognitive processes of unseen statistic maps from large collections of the NeuroVault database. We show that such a self-taught learning process always improves the performance of the classifiers but the magnitude of the benefits strongly depends on the number of data available both for pre-training and finetuning the models and on the complexity of the targeted downstream task.
  
## Table of contents
   * [How to cite?](#how-to-cite)
   * [Contents overview](#contents-overview)
   * [Reproducing figures and tables](#reproducing-figures-and-tables)
      * [Installing environment](#environment)
      * [Download necessary data](#download-data)
      * [Table 1](#table-1)
      * [Table 2](#table-2)
      * [Figure 3](#fig-3)
      * [Figure 4](#fig-4)
      * [Figure 5](#fig-5)
      * [Figure 6](#fig-6)
      * [Figure 7](#fig-7)
   * [Reproducing full analysis](#reproducing-full-analysis)

## How to cite?


# Contents overview


## `src`

This directory contains scripts and notebooks used to launch the full analysis. 
Two notebooks (`dataset_selection_notebook.ipynb` and `download_and_preprocess_data_notebook.ipynb`) contains the functions used to treat NeuroVault metadata, select necessary data, download and preprocess them.

## `data`

This directory is made to contain data that will be used by scripts/notebooks stored in the `src` directory and to contain results of those scripts. 

## `results`

This directory contains notebooks and scripts that were used to analyze the results of the experiments. These notebooks were used to evaluate the performance of autoencoders and classifiers and to compare these performance in different settings. 

## `figures`

This directory contains figures and csv files obtained when running the notebooks in the `results` directory.

## Reproducing figures and tables

### Installing environment
To reproduce the figures, you will need to create a conda environment with the necessary packages. 

- First, download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [miniforce](https://github.com/conda-forge/miniforge) if you work on a Mac Apple Silicon. 
- After this, you will need to create you own environment by running : 
```
conda create -n workEnv
conda activate workEnv
```
The name workEnv can be changed. 
- When you environment is activated, just run the `install_environment.sh` script available at the root of this directory. 
```
bash install_environment.sh
```
This script will install all the necessary packages to reproduce this analysis. 
At each step, you might have to answer y/n, answer yes any time to properly do the install. 

### Download necessary data 
Using the notebook `src/download_and_preprocess_data_notebook.ipynb`, necessary data will be downloaded and preprocessed. 

### Table 1
**Overview of the datasets. For each dataset, number of statistic maps are presented, as well as the number of subjects, number of studies and the type of labels (if available).**

To reproduce the results presented in Table 1, use the notebook `src/dataset_selection_notebook.ipynb`. Place yourself at the root of the directory and run the command:

```
jupyter notebook src/dataset_selection_notebook.ipynb
```

In this notebook, the different datasets are selected and split in train/test/valid if necessary. 
Since the selection is randomised for some datasets, we recommand to keep the files stored in the `data` repository and to launch the notebook. The corresponding values will be printed at each step.

### Table 2
**Reconstruction performance of the CAE depending on model architecture and training set. Values are the mean Pearson's correlation coefficients (standard error of the mean).**

To reproduce the results presented in Table 2, use the notebook `results/ae_reconstruction_notebook.ipynb`. 
Place yourself at the root of the directory and run the command:

```
jupyter notebook results/ae_reconstruction_notebook.ipynb
```

In this notebook, we first analyse the performance of the different autoencoder trained (4 and 5 layers architectures). 
When these performance are computed, mean values are computed.

### Figure 3
**Original version and reconstruction of a randomly drawn statistic map of NeuroVault test dataset (image ID: 109) with the two CAEs (4-layers and 5-layers).**

To reproduce the results presented in Table 2, use the notebook `results/ae_reconstruction_notebook.ipynb`.

Place yourself at the root of the directory and run the command:

```
jupyter notebook results/ae_reconstruction_notebook.ipynb
```

In this notebook, we first analyse the performance of the different autoencoder trained (4 and 5 layers architectures). 
When these performance are computed, we plot the original and reconstructed version of an example map.

### Table 3 
**Hyperparameters chosen for each dataset and corresponding performance of the classifier on the validation set of the dataset**

Results presented in Table 3 can be reproduced using `hcp_dataset_notebook.ipynb` and `brainpedia_dataset_notebook.ipynb`. 

Place yourself at the root of the directory and run the command:

```
jupyter notebook results/hcp_dataset_notebook.ipynb
jupyter notebook results/brainpedia_dataset_notebook.ipynb
```

In the first one, we analyze the performance of different model with different hyperparameters on HCP dataset and search for the best models. In the second one, we use BrainPedia dataset. Run the cells at the beginning of the notebook until you arrive at *Hyperparameter validation - Exploration of results*. In the first cells, you will compute the performance of the different classifiers and then, explore these results.

### Table 4
**Classification performance on HCP datasets of models initialized with default algorithm vs with the weights of a pre-trained CAE. Values are described as the average (standard error) of the metric on the 5 fold of cross-validation. Paired two-sample t-tests results between the performance of the pre-trained classifier versus the default algorithm initialization one are indicated above. DA: Default Algorithm initialization ; PT: Pre-Training initialization.** 


Results presented in Table 4 can be reproduced using `hcp_dataset_notebook.ipynb`. 

Place yourself at the root of the directory and run the command:

```
jupyter notebook results/hcp_dataset_notebook.ipynb
```

Here, we analyze the performance of the models trained on a cross-validation framework on HCP datasets with different sample sizes and different tasks. Run the cells after *Test models with cross-validation*.  

### Figure 4

**Mean accuracy (and standard error) on \underline{contrast classification} with the HCP dataset for the models initialized with default algorithm (blue) and pre-trained CAE (orange). Pre-training improves contrast classification performance for small sample sizes and at a lower level of improvement, also for large sample sizes.**

Results presented in Figure 4 can be reproduced using `hcp_dataset_notebook.ipynb`. 

Place yourself at the root of the directory and run the command:

```
jupyter notebook results/hcp_dataset_notebook.ipynb
```

Here, we analyze the performance of the models trained on a cross-validation framework on HCP datasets with different sample sizes and different tasks. Run the cells after *Test models with cross-validation*. 

### Figure 5

**Mean accuracy (and standard error) on \underline{task classification} with the HCP dataset for the models initialized with default algorithm (blue) and pre-trained CAE (orange). Pre-training improves task classification performance for all sample sizes but sample sizes does not have a huge influence on the level of improvement.**

Results presented in Figure 5 can be reproduced using `hcp_dataset_notebook.ipynb`. 

Place yourself at the root of the directory and run the command:

```
jupyter notebook results/hcp_dataset_notebook.ipynb
```

Here, we analyze the performance of the models trained on a cross-validation framework on HCP datasets with different sample sizes and different tasks. Run the cells after *Test models with cross-validation*. 

### Figure 6

**Mean accuracy (and standard error) on one contrast task classification with the HCP dataset for the models initialized with default algorithm (blue) and pre-trained CAE (orange). Pre-training does not always improve one-contrast task classification performance: for some sample sizes, pre-training and default initialization give very similar results.**

Results presented in Figure 6 can be reproduced using `hcp_dataset_notebook.ipynb`. 

Place yourself at the root of the directory and run the command:

```
jupyter notebook results/hcp_dataset_notebook.ipynb
```

Here, we analyze the performance of the models trained on a cross-validation framework on HCP datasets with different sample sizes and different tasks. Run the cells after *Test models with cross-validation*. 

### Table 5

**Classification performances on BrainPedia datasets of models initialized with default algorithm vs with the weights of a pre-trained CAE. Values are described as the average (+/- standard error) of the metric on the 5 fold of cross-validation. Paired two-sample t-tests results between the performance of the pre-trained classifier versus the default algorithm initialization one are indicated above. DA: Default Algorithm initialization ; PT: Pre-Training initialization**

Results presented in Table 5 can be reproduced using `brainpedia_dataset_notebook.ipynb`. 

Place yourself at the root of the directory and run the command:

```
jupyter notebook results/brainpedia_dataset_notebook.ipynb
```

Here, we analyze the performance of the models trained on a cross-validation framework on BrainPedia datasets. Run the cells after *Test models with cross-validation*. 

### Figure 7

**Mean F1-score (and standard error) of the classification of mental concepts on BrainPedia datasets (Small and Large) for the models initialized with default algorithm (blue) and pre-trained CAE (orange). Pre-training improves classification performance, in particular for the small dataset.**

Results presented in Table 5 can be reproduced using `brainpedia_dataset_notebook.ipynb`. 

Place yourself at the root of the directory and run the command:

```
jupyter notebook results/brainpedia_dataset_notebook.ipynb
```

Here, we analyze the performance of the models trained on a cross-validation framework on BrainPedia datasets. Run the cells after *Test models with cross-validation*. 

## Reproducing full analysis

First, see [Install environment](#environment) and [Download necessary data](#download-data) to prepare the environment and necessary data. 

### CAE training 

To launch the training of the AutoEncoder on NeuroVault dataset, use the `autoencoder_training.py` script. 

```
python autoencoder_training.py -d ../data/preprocessed/NeuroVault_dataset -o ../data/derived -e 200 -b 32 -p resampled_masked_normalized -s neurovault_dataset -m model_cnn_4layers -l 1e-04
```

Here, the command line launch the training of the autoencoder for 200 epochs (`-e 200`), a batch size of 32 (`-b 32`), a learning rate of 1e-04 (`-l 1e-04`) and the model 4-layers (`-m model_cnn_4layers`). These options can be modified and in particular, the model option to launch training with model 5-layers: `-m model_cnn_5layers`.

### CNN training and fine-tuning

To launch the training of the CNN on the different datasets, use the `cnn_training.py` script.

For example, here is the commandline to launch the training of the CNN from scratch on HCP global dataset.
```
python cnn_training.py -d ../data/preprocessed/HCP_dataset -o ../data/derived -e 500 -b 32 -p resampled_masked_normalized -s hcp_global_dataset -m model_cnn_4layers -l 1e-04 -r kfold -c contrast
```

Different options can be modified. 
- Number of epochs: `-e`
- Batch size: `-b`
- Dataset: `-s`
- Model: `-m` with `model_cnn_4layers` or `model_cnn_5layers`
- Learning rate: `-l`
- Method of training: `-r` with 
    - `kfold` for a training from scratch (default initialization) and a 5-fold cross-validation 
    - `no` for a training from scratch (default initialization) without cross-validation
    - `all_kfold` for a training with the weights initialized with pre-trained AutoEncoder and a 5-fold cross-validation
    - `all` for a training with the weights initialized with pre-trained AutoEncoder without cross-validation

