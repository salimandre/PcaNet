# PcaNet applied to face classification

We implemented the PCA network described in following reference:

> *PCANet: A Simple Deep Learning Baseline for Image Classification?*, Tsung-Han Chan, Kui Jia, Shenghua Gao, Jiwen Lu, Zinan Zeng, and Yi Ma, 2014

We evaluated the PcaNet model on **Labeled Faces in the Wild** (LFW) dataset using an SVM as a classifier.

## Summary of the method

* PcaNet is a model which computes a **cascade of convolutions**. At each level of the network **filters** are learnt using PCA. It is achieved by first **sampling patches** from dataset. Then PCA learns an **orthogonal basis of patches** which allows to decompose any patches in this new basis so that it looses the less information. 

* A **binary hashing step** binarize outputs. For each pixel it provides a binary sequence, hence an integer.

* Finally hashed outputs are encoded by **blocks of histograms**

For a layers PcaNet model we have the following number of features:

N_Features = N_filters_layer_1 * N_Blocks * 2^N_filters_layer_2
