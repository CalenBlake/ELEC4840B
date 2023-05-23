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
data_dir = '../RAVDESS_Refactored/RGB_IMG/'
dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())

# b.) separate into k-fold sets %%%%%%%%%%


