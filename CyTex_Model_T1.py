"""
# -----------------------------------
# Construction of the DCNN model using the ResNet 50 model as a baseline.
# Transfer learning approach.
#
#
# Author: Calen Blake
# Date: 01-04-23
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
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from keras.utils import plot_model
import visualkeras
from torchvision.utils import save_image
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import time
import random
import os

# --------------------- 1. Load & Preprocess Data ---------------------
# NOTE: Will need to sort data into training and test sets. Check how this was done in ENGG3300.
# This can either be done in code or can make train and val folders in the data directory.
# a.) load data %%%%%%%%%%
batch_size = 32
img_height = 400
img_width = 400

# Plot to separate window and remove Matplotlib version 3.6 warnings
matplotlib.use("TkAgg")

# Define training transforms
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Resize(256),
    # Mean and std values from ImageNet benchmark dataset
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Define test transforms -> No image altering
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(256),
    # Mean and std values from ImageNet benchmark dataset
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load train and test datasets
train_dir = "./EMODB Database/RGB_IMG_Split/train/"
test_dir = "./EMODB Database/RGB_IMG_Split/test/"
# Reload the newly created training and testing datasets, applying transforms
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# Load train and test data using dataloader
# Note: A dataloader wraps an iterable around the Dataset to enable easy access to the samples,
# passes data through in batches of specified size
train_loader = data.DataLoader(
    train_dataset,
    # Tune batch_size later for better performance
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    # set pin_memory=True to enable fast data transfer to CUDA-enabled GPUs
    pin_memory=False
)
test_loader = data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    # Preserve original order of test samples to evaluate predictions consistently
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

# e.) plot sample of transformed training data %%%%%%%%%%
# Get a random batch of images and labels
t_images, labels = next(iter(train_loader))
# Plot a sample of 6 images from the batch
fig, axs = plt.subplots(2, 3, figsize=(12, 6))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(t_images[i].permute(1, 2, 0))
    class_name = train_dataset.classes[labels[i]]
    ax.set_title(f"Class: {class_name}")
plt.suptitle('Random Sample of Training Data')
plt.tight_layout()
plt.show()

# --------------------- 2. Construct Model - ResNet50 ---------------------
# NOTE: Need to add additional layers on the output of the ResNet model from the CyTex academic paper.
# This will yield 7 different output classes, 1 for each of the emotional classes.

# Load ResNet50 model, pretrained being depreciated, instead specify weights
# Access weights at: https://pytorch.org/vision/stable/models.html
# OLD SYNTAX: model_rn50 = models.resnet50(pretrained=True)
model_rn50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# FREEZE Weights of the first two blocks, retrain remaining by default
# named_parameters() returns (str, Parameter) – Tuple containing the name and parameter
# Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
for name, param in model_rn50.named_parameters():
    if 'layer1' in name or 'layer2' in name:
        # Uncomment to see names of layers and contents
        # print(name)
        param.requires_grad = False

# Alternative weight freezing method highlighted in the following link:
# https://discuss.pytorch.org/t/model-named-parameters-will-lose-some-layer-modules/14588

# get number of input features for the last fully connected layer
# This will feed into the following layers we create at the output of the ResNet model
num_features = model_rn50.fc.in_features
print('\n--------------------')
print('input features of final resnet50 layer (Pre CyTex additions):')
print(num_features)
print('--------------------')
# num_features = 2048

# View the architecture of the ResNet50 model
# NOTE: Can use "from keras.utils import plot_model" to plot the model architecture
# Alternatively: "visualkeras.layered_view(<model>)"
print('\n--------------------')
print('Pre-modification model architecture:')
# print(model_rn50)
print('--------------------')

# Check shape of example data
print('\n--------------------')
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
print('The shape of the example data is:')
print(example_data.shape)
# returns: torch.Size([32, 3, 400, 400])
print('--------------------')

# Use sequential to add layers -> Same as described in Ali's paper
# !!!RESEARCH: Get detailed summary of the function of each of the layers in the network + further research
model_rn50.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Flatten(),
    nn.Linear(num_features, 4096),
    nn.BatchNorm1d(4096),
    nn.Dropout(p=0.55),
    nn.Linear(4096, 1024),
    nn.BatchNorm1d(1024),
    # ASK ALI: Should modify shape to be num of classes!?
    nn.Linear(1024, len(train_dataset.classes)),
    # dim=1 applies the softmax operation over the first dimension of the tensor
    # Returns probability distribution over the classes for each example in each batch
    nn.Softmax(dim=1)
)

# View new model architecture
print('\n--------------------')
print('New modified model architecture:')
# print(model_rn50)
print('--------------------')



# Pass test tensor through model to check shape
output_tensor = model_rn50(example_data)
print('\n--------------------')
print('The shape of the example data, after passing through the model, is:')
print(output_tensor.shape)
print('--------------------')

# Enable cuda (if available)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_ft = model_rn50.to(device)


# --------------------- 3. Train & Evaluate Model ---------------------
# Initially train for minimal epochs and check results then scale up to ~200 epochs

# Use following line to check time of each epoch iteration...
# since = time.time()

# # train your model on the training data
# for epoch in range(num_epochs):
#     for images, labels in train_loader:
#         # train your model here
#     for images, labels in test_loader:
#         # evaluate your model here

# Plot the train and test loss (exp dec)
# Plot the train and test acc (exp inc)

# --------------------- 4. Save Params & Visualize Results ---------------------


