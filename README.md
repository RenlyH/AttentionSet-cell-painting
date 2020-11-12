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

The code can be used to run the Noise-signal mixture experiment.

How to Use
----------
`data_loader.py`: Generates training and test set by combining segmented cell images to bags. A bag is given a positive label if it contains one or more images sampled from treatment dataset.

`base_exp.py`: Baseline model 

`main.py`: Trains a small CNN with the Adam optimization algorithm.
The training takes 20 epochs. Last, the accuracy and loss of the model on the test set is computed.
In addition, a subset of the bags labels and instance labels are printed.

`model.py`: The model is a modified LeNet-5, see <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>.
The Attention-based MIL pooling is located before the last layer of the model.
The objective function is the negative log-likelihood of the Bernoulli distribution.

