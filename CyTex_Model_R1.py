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
# a.) load data %%%%%%%%%%
batch_size = 32
img_height = 400
img_width = 400

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
train_dir = "./RAVDESS_Refactored/RGB_IMG_Split/train/"
test_dir = "./RAVDESS_Refactored/RGB_IMG_Split/test/"
# Reload the newly created training and testing datasets, applying transforms
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# Load train and test data using dataloader
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

# ----- Can plot images of training data pre and post augmentation here -----

# --------------------- 2. Construct Model - ResNet50 ---------------------
# Load ResNet50 model, pretrained being depreciated, instead specify weights
# Access weights at: https://pytorch.org/vision/stable/models.html
model_rn50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# FREEZE Weights of the first two blocks, retrain remaining by default
for name, param in model_rn50.named_parameters():
    if 'layer1' in name or 'layer2' in name:
        param.requires_grad = False

# get number of input features for the last fully connected layer
num_features = model_rn50.fc.in_features
# Add regularization layers to the output of the TL model => Improve Generalization
model_rn50.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Flatten(),
    nn.Linear(num_features, 4096),
    nn.BatchNorm1d(4096),
    nn.Dropout(p=0.55),
    nn.Linear(4096, 1024),
    nn.BatchNorm1d(1024),
    nn.Linear(1024, len(train_dataset.classes)),
    # dim=1 applies the softmax operation over the first dimension of the tensor
    # Returns probability distribution over the classes for each example in each batch
    nn.Softmax(dim=1)
)

# --------------------- 3. Train & Evaluate Model ---------------------
# a.) Set device (Cuda or CPU) %%%%%%%%%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_rn50 = model_rn50.to(device)

n_epochs = 60
n_batches = np.ceil(len(train_dataset)/batch_size)

# b.) Print some useful info before training
print('\n--------------------')
print('Total data samples: ', len(train_dataset) + len(test_dataset))
print('Train data samples: ', len(train_dataset))
print('Test data samples: ', len(test_dataset))
print(f'batch size: {batch_size:.0f} --> training batches: {n_batches:.0f}')
print(f'epochs: {n_epochs:.0f} --> total batches: {(n_epochs*n_batches):.0f}')
print('--------------------')

# c.) Define loss function, optimizer, lr scheduler and run-time stats %%%%%%%%%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_rn50.parameters(), lr=1e-2, weight_decay=1e-5)
# Decay LR by a factor of 0.1 every [step_size] epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.01)
train_loss = []
test_loss = []
train_acc = []
test_acc = []

# d.) Create callable functions for model training & testing %%%%%%%%%%
def train_model(model, criterion, optimizer, scheduler):
    model.train()

    running_loss = 0.0
    running_corrects = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # FORWARD ----------
        # zero gradients
        optimizer.zero_grad()
        # forward pass inputs through the model
        outputs = model(inputs)
        # predictions are the class with the highest prediction probability
        _, preds = torch.max(outputs, 1)
        # calculate the loss
        loss = criterion(outputs, labels)
        # back-propagate the loss
        loss.backward()
        # update the model parameters
        optimizer.step()
        # calculate running loss and acc
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()
        # FORWARD END ----------
    # step the scheduler on an epoch passing basis!
    scheduler.step()
    # calculate + print: loss and acc over epoch_i
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100 * running_corrects / len(train_dataset)
    print(f'train loss: {epoch_loss}, train acc: {epoch_acc}')
    # append epoch results to corresponding lists
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)


def test_model(model):
    model.eval()

    running_loss = 0.0
    running_corrects = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # FORWARD ----------
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
            # FORWARD END ----------
        # calculate + print: loss and acc over epoch_i
        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = 100 * running_corrects / len(test_dataset)
        print(f'test loss: {epoch_loss}, test acc: {epoch_acc}')
        # append epoch results to corresponding lists
        test_loss.append(epoch_loss)
        test_acc.append(epoch_acc)


# e.) Execute model training and testing %%%%%%%%%%
print('\nINITIATING MODEL TRAINING & TESTING...')
print('-' * 10)
since = time.time()
for epoch_i in range(n_epochs):
    # since_epoch = time.time()
    print(f'Epoch {epoch_i + 1}/{n_epochs}')
    # TRAINING + Display epoch stats
    train_model(model_rn50, criterion, optimizer, exp_lr_scheduler)
    # TESTING + Display epoch stats
    test_model(model_rn50)
    print('-' * 10)
# Print total time of training + testing
time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print('-' * 10)
print()

# f.i.) Plot the train and test loss (exp dec) %%%%%%%%%%
plt.figure()

plt.subplot(1, 2, 1)
plt.plot(list(range(1, n_epochs+1)), train_loss, 'b')
plt.plot(list(range(1, n_epochs+1)), test_loss, 'r')
plt.title('Simulation Loss')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('N epochs')
plt.ylabel('Average loss')
plt.grid()

# f.ii.) Plot the train and test acc (exp inc) %%%%%%%%%%
plt.subplot(1, 2, 2)
plt.plot(list(range(1, n_epochs+1)), train_acc, 'b')
plt.plot(list(range(1, n_epochs+1)), test_acc, 'r')
plt.title('Prediction Accuracy')
plt.legend(['Train Acc', 'Test Acc'], loc='upper right')
plt.xlabel('N epochs')
plt.ylabel('Model accuracy (%)')
plt.grid()

plt.tight_layout()
plt.show()

# --------------------- 4. Save & Load Params ---------------------
# a.) Save trained model parameters %%%%%%%%%%
timestamp = datetime.datetime.now().strftime("%d-%m__%H-%M")
filename = f"model_params_{timestamp}.pt"
torch.save(model_rn50.state_dict(), f'rn50 saved params - RAVDESS/{filename}')
"""
# b.) Load trained model parameters %%%%%%%%%%
model = model_rn50
load_file = f'rn50 saved params - RAVDESS/'
model.load_state_dict(torch.load(load_file))
"""
