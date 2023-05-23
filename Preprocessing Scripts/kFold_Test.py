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

# a.) define transforms %%%%%%%%%%
# Define training transforms
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    # Mean and std values from ImageNet benchmark dataset
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Define test transforms -> No image altering
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# b.) load data %%%%%%%%%%
data_dir = '../EMODB Database/RGB_IMG/'
dataset = datasets.ImageFolder(data_dir)
# only known way to apply transforms independently!!!
dataset_train = datasets.ImageFolder(data_dir, transform=train_transforms)
dataset_test = datasets.ImageFolder(data_dir, transform=test_transforms)

# c.) testing subset method %%%%%%%%%%
# x = dataset.imgs    # directory paths to images
# sub_d = data.Subset(dataset, range(int(len(dataset)/2)))
# sub_d2 = data.Subset(dataset, [0, 1, 2])

# d.) create k-fold splits %%%%%%%%%%
y = dataset.targets     # list of labels
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
splits = skf.split(range(len(y)), y)



# e.) split data into different folds and alternate %%%%%%%%%%
# Setup dummy var x (only used for length, not value)
x = np.zeros(len(y))
# generate split indices and print to test if it works! -> commented out right now
for fold, (train_indices, test_indices) in enumerate(skf.split(x, y)):
    print(f"Training on fold {fold + 1}/{k}")
    # print(test_indices)
    train_dataset = data.Subset(dataset_train, train_indices)
    test_dataset = data.Subset(dataset_test, test_indices)

