# RMFFN
## Introduction
This is the official implementation of the paper ["Reinforcement Learning Based Multi-modal Feature Fusion Network for Novel Class Discovery"].

## Abstract

With the development of deep learning techniques, supervised learning has achieved performances surpassing those of humans. Researchers have designed numerous corresponding models for different data modalities, achieving excellent results in supervised tasks. However, with the exponential increase of data in multiple fields, the recognition and classification of unlabeled data have gradually become a hot topic. In this paper, we employed a Reinforcement Learning framework to simulate the cognitive processes of humans for effectively addressing novel class discovery in the Open-set domain. We deployed a Member-to-Leader Multi-Agent framework to extract and fuse features from multi-modal information, aiming to acquire a more comprehensive understanding of the feature space. Furthermore, this approach facilitated the incorporation of self-supervised learning to enhance model training. We employed a clustering method with varying constraint conditions, ranging from strict to loose, allowing for the generation of dependable labels for a subset of unlabeled data during the training phase. This iterative process is similar to human exploratory learning of unknown data. These mechanisms collectively update the network parameters based on rewards received from environmental feedback. This process enables effective control over the extent of exploration learning, ensuring the accuracy of learning in unknown data categories. We demonstrate the performance of our approach in both the 3D and 2D domains by employing the OS-MN40, OS-MN40-Miss, and Cifar10 datasets. Our approach achieves competitive competitive results.

## Quick start
Install cuda, PyTorch and torchvision.
Please make sure they are compatible. We test our models on:
'''cuda==11, torch==1.9.0, torchvision==0.10.0, python==3.7.3'''
## Data preparation
Download [OS-MN40 & OS-MN40-Miss](https://www.moon-lab.tech/shrec22). \n
Make sure the data set file is stored as the example below:"/Target/Bathtub/Bathtub0001/img/1.png"

Obtaion the label embedding calculated by [Query2label](https://arxiv.org/abs/2107.10834).

## Train the model
### Pretrain
Use pretrain.py to pretrain the model. After pretraining, you need to run a DBSCAN clustering to obtain the initial hyper-parameters.

### Train

Use main.py to train the model.


