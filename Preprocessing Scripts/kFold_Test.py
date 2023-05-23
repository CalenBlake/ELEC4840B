"""
# -----------------------------------
# Load dataset and test separating into k folds...
#
#
# Author: Calen Blake
# Date: 23-05-23
# NOTE: Currently not included in GitHub and just used for testing purposes
# -----------------------------------
"""

# --------------------- Import necessary libraries ---------------------
import torch.utils.data as data
import numpy as np
import torchvision
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import random
import os
from sklearn.model_selection import StratifiedKFold

# a.) load data %%%%%%%%%%
data_dir = '../EMODB Database/RGB_IMG/'
dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())

# b.) testing subset method %%%%%%%%%%
x = dataset.imgs    # directory paths to images
sub_d = data.Subset(dataset, range(int(len(dataset)/2)))
sub_d2 = data.Subset(dataset, [0, 1, 2])

# c.) create k-fold splits %%%%%%%%%%
y = dataset.targets     # list of labels
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = skf.split(range(len(y)), y)
for train_index, test_index in splits:
    X_train, X_test = [dataset[i][0] for i in train_index], [dataset[i][0] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

