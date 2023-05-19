"""
# -----------------------------------
# Construction of the DCNN model using the ResNet 50 model as a baseline.
# Transfer learning approach. For analysing RAVDESS database.
# Results should be notably better than initial EMODB results due to
# A larger database and fewer possible sentences
#
# Author: Calen Blake
# Date: 19-05-23
# NOTE: Could truncate # emotions, trial different CyTex outputs to improve
#       results, k-fold may not be necessary here.
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
from torchvision.utils import save_image
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
import time
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# --------------------- 1. Load & Preprocess Data ---------------------
