#/bin/bash

conda install -c anaconda jupyter
conda install pandas
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge nibabel
conda install -c conda-forge matplotlib
conda install -c conda-forge nilearn
conda install -c conda-forge tensorboard
conda install -c anaconda scikit-learn