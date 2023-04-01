"""
# -----------------------------------
# Construction of the DCNN model using the ResNet 50 model as a baseline.
# Transfer learning approach
# The model will be used to classify images of ants and bees.
#
# Author: Calen Blake
# Date: 25-03-23
# NOTE: This script should later be divided into separate modules.
# i.e. model, main, save_params, load_params, test model... as was done in other research papers (see TIM_Net)
# Or as was done in ENGG3300!
# -----------------------------------
"""

# --------------------- Import necessary libraries ---------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
# import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# import time
import random
import os

# --------------------- 1. Load & Preprocess Data ---------------------
# NOTE: Will need to sort data into training and test sets. Check how this was done in ENGG3300.
# This can either be done in code or can make train and val folders in the data directory.
# a.) load data
data_dir = './EMODB Database/RGB_IMG/'
batch_size = 32
img_height = 400
img_width = 400

# import full dataset -> load from folders
dataset = datasets.ImageFolder(data_dir)

# b.) plot sample of data
# Get a random subset of images from the dataset
indices = random.sample(range(len(dataset)), 6)
subset = [dataset[i] for i in indices]
# Plot to separate window and remove Matplotlib version 3.6 warnings
matplotlib.use("TkAgg")
# Plot the images with classes as titles
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
plt.title('Random Sample of Input Data')
for i, ax in enumerate(axes.flat):
    ax.imshow(subset[i][0])
    ax.set_title(dataset.classes[subset[i][1]])
    # ax.axis('off')
plt.suptitle('Random Sample of Input Data')
plt.tight_layout()
plt.show()



# c.) separate into training and test sets and apply transforms
# split data into training and test set, 80/20 split
# train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
# reload data using dataloader objects -> apply transforms to train and test sets


# d.) plot sample of transformed training data


# --------------------- 2. Construct Model - ResNet50 ---------------------
# NOTE: Need to add additional layers on the output of the ResNet model from the CyTex academic paper.
# This will yield 7 different output classes, 1 for each of the emotional classes.

# # train your model on the training data
# for epoch in range(num_epochs):
#     for images, labels in train_loader:
#         # train your model here
#     for images, labels in test_loader:
#         # evaluate your model here

# --------------------- 3. Train & Evaluate Model ---------------------
# Initially train for minimal epochs and check results then scale up to ~200 epochs

# Plot the train and test loss (exp dec)
# Plot the train and test acc (exp inc)

# --------------------- 4. Save Params & Visualize Results ---------------------


