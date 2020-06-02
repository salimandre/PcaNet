# PcaNet applied to face classification

We implemented the PCA network described in following reference:

> *PCANet: A Simple Deep Learning Baseline for Image Classification?*, Tsung-Han Chan, Kui Jia, Shenghua Gao, Jiwen Lu, Zinan Zeng, and Yi Ma, 2014

We evaluated the PcaNet model on **Labeled Faces in the Wild** (LFW) dataset using an SVM as a classifier.

## Summary of the method

* PcaNet is a model which computes a **cascade of convolutions**. At each level of the network **filters** are learnt using PCA. It is achieved by first **sampling patches** from dataset. Then PCA learns an **orthogonal basis of patches** which allows to decompose any patches in this new basis so that it looses the less information. 

* A **binary hashing step** binarize outputs. For each pixel it provides a binary sequence, hence an integer.

* Finally hashed outputs are encoded by **blocks of histograms**

* For a 2 layers PcaNet model we have the following number of features:

N_Features = N_filters_layer_1 * N_Blocks * 2^N_filters_layer_2


## LFW dataset

* We used LFW dataset: 1288 images of faces from 7 political leaders (Ariel Sharon, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Hugo Chavez, Tony Blair). Dataset contains at least 70 images of each politician of size (50, 37).
* We made a 75/25 split of the dataset in order to get train and test sets

## Our results

We could achieve 97% accuracy on LFW dataset. LFA dataset





