AttentionSet in image-based high-content screening
================================================

by Renly Xinhai Hou (<renly.hou@gmail.com>)

Overview
--------

PyTorch implementation of our paper "AttentionSet in image-based high-content screening".


Installation
------------

Installing Pytorch 0.3.1, using pip or conda, should resolve all dependencies.
Tested with Python 3.7.6, but should work with 3.x as well.
Tested on both CPU and GPU.

Content
--------

The code can be used to run the Noise-signal mixture experiment with baseline model and Set-based model.

How to Use
----------
`data_loader.py`: Generates training and test set by combining segmented cell images to bags. A bag is given a positive label if it contains one or more images sampled from treatment dataset.

`base_exp.py`: Baseline model for noise-signal mixture experiment. 

`set_experiment.py`: Prediction based on set for noise-signal mixture experiment.

`model.py`: It consists of DeepSet model and AttentionSet model for a batch of profile input. 
The AttentionSet pooling method is located before the last layer of the model.

`main.py`: Trains a small DeepSet model with the Adam optimization algorithm.
The training takes 40 epochs. Last, the accuracy and loss of the model on the test set is computed.

`main_att.py`: Trains a small DeepSet model with the Adam optimization algorithm.
The training takes 40 epochs. Last, the accuracy and loss of the model on the test set is computed.
