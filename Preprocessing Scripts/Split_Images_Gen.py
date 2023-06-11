"""
# -----------------------------------
# For taking the RGB CyTex images as input and splitting the images into
# train and test sets. Then saving the output into separate file streams.
#
# Author: Calen Blake
# Date: 03-04-23
# NOTE:
# -----------------------------------
"""

# --------------------- Import necessary libraries ---------------------
import torch.utils.data as data
# import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
import matplotlib
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# import time
import random
import os

# --------------------- 1. Load & Preprocess Data ---------------------
# NOTE: Will need to sort data into training and test sets. Check how this was done in ENGG3300.
# This can either be done in code or can make train and val folders in the data directory.
# a.) load data %%%%%%%%%%
data_dir = '../RAVDESS_Refactored/RGB_IMG_Culled/'
batch_size = 32
img_height = 400
img_width = 400
# import full dataset -> load from folders, don't convert to tensor to enable plotting
dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())

# b.) plot sample of data %%%%%%%%%%
# Plot to separate window and remove Matplotlib version 3.6 warnings
# matplotlib.use("TkAgg")

# # Get a random sample of 6 images and their classes
# indices = np.random.choice(len(dataset), size=6, replace=False)
# images = [dataset[i][0] for i in indices]
# labels = [dataset.classes[dataset[i][1]] for i in indices]

# Plot the images with classes as titles
# fig, axes = plt.subplots(2, 3, figsize=(12, 6))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(images[i].permute(1, 2, 0))
#     # ALI CHECK: Please check that the correct classes are displayed alongside the images
#     ax.set_title(f"Class: {labels[i]}")
# plt.suptitle('Random Sample of Input Data')
# plt.tight_layout()
# plt.show()

# c.) separate into training and test sets %%%%%%%%%%
# split data into training and test set
train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)
train_data = data.Subset(dataset, train_indices)
test_data = data.Subset(dataset, test_indices)

# e.) save subsets in directories and reload %%%%%%%%%%
# This will maintain the properties of a dataset. For instance, 'Subset' object has no attribute 'classes'
# Create a new ImageFolder object for the train subset
train_dir = "../RAVDESS_Refactored/RGB_IMG_Split/train/"
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
test_dir = "../RAVDESS_Refactored/RGB_IMG_Split/test/"
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
