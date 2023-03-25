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
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import os

# --------------------- 1. Load & Preprocess Data ---------------------
# NOTE: Will need to sort data into training and test sets. Check how this was done in ENGG3300.
# This can either be done in code or can make train and val folders in the data directory.


# --------------------- 2. Construct Model - Feature Extraction ---------------------
# NOTE: Need to add additional layers on the output of the ResNet model from the CyTex academic paper.
# This will yield 7 different output classes, 1 for each of the emotional classes.

# --------------------- 3. Train & Evaluate Model ---------------------


# --------------------- 4. Save Params & Visualize Results ---------------------


