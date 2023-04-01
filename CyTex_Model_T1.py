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
from torchvision.utils import save_image
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
# a.) load data %%%%%%%%%%
data_dir = './EMODB Database/RGB_IMG/'
batch_size = 32
img_height = 400
img_width = 400
# import full dataset -> load from folders, don't convert to tensor to enable plotting
dataset = datasets.ImageFolder(data_dir)

# b.) plot sample of data %%%%%%%%%%
# Get a random subset of images from the dataset
indices = random.sample(range(len(dataset)), 6)
subset = [dataset[i] for i in indices]
# Plot to separate window and remove Matplotlib version 3.6 warnings
matplotlib.use("TkAgg")
# Plot the images with classes as titles
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(subset[i][0])
    ax.set_title(dataset.classes[subset[i][1]])
    # ax.axis('off')
plt.suptitle('Random Sample of Input Data')
plt.tight_layout()
plt.show()

# c.) separate into training and test sets and apply transforms %%%%%%%%%%
# Define training transforms
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    # Mean and std values from ImageNet benchmark dataset
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Define test transforms -> No image altering
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    # Mean and std values from ImageNet benchmark dataset
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# split data into training and test set, 80/20 split
train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
train_data = data.Subset(dataset, train_indices)
test_data = data.Subset(dataset, test_indices)

# e.) save subsets in directories and reload %%%%%%%%%%
# This will maintain the properties of a dataset. For instance, 'Subset' object has no attribute 'classes'
# Create a new ImageFolder object for the train subset
train_dir = "./EMODB Database/RGB_IMG/train/"
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
for i in range(len(train_data)):
    image, label = train_data[i]
    class_dir = os.path.join(train_dir, dataset.classes[label])
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    filename = f"{i}.png"
    save_path = os.path.join(class_dir, filename)
    save_image(image, save_path)

# Create a new ImageFolder object for the test subset
test_dir = "./EMODB Database/RGB_IMG/test/"
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
for i in range(len(test_data)):
    image, label = test_data[i]
    class_dir = os.path.join(test_dir, dataset.classes[label])
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    filename = f"{i}.png"
    save_path = os.path.join(class_dir, filename)
    save_image(image, save_path)

# Reload the newly created training and testing datasets
train_data = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
# Check to see whether the classes are an attribute -> correctly saved and loaded dataset
print(train_data.classes)
test_data = datasets.ImageFolder(test_dir, transform=transforms.ToTensor())
print(test_data.classes)

# Apply the transforms defined above to the train and test data
train_data.dataset.transform = train_transforms
test_data.dataset.transform = test_transforms
# Load train and test data using dataloader
# Note: A dataloader wraps an iterable around the Dataset to enable easy access to the samples,
# passes data through in batches of specified size
train_loader = data.DataLoader(
    train_data,
    # Tune batch_size later for better performance
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    # set pin_memory=True to enable fast data transfer to CUDA-enabled GPUs
    pin_memory=False
)
test_loader = data.DataLoader(
    test_data,
    batch_size=1000,
    # Preserve original order of test samples to evaluate predictions consistently
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

# e.) plot sample of transformed training data %%%%%%%%%%


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


