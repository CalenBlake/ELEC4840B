"""
# -----------------------------------
# Construction of the DCNN model using the ResNet 50 model as a baseline.
# Transfer learning approach. For analysing EMODB database.
#
#
# Author: Calen Blake
# Date: 01-04-23
# NOTE: This script should later be divided into separate modules.
#       i.e. model, main, save_params, load_params, test model... as was done in other research papers (see TIM_Net)
#       Or as was done in ENGG3300!
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
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
import time
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from sklearn.model_selection import train_test_split

# --------------------- 1. Load & Preprocess Data ---------------------
# a.) load data %%%%%%%%%%
batch_size = 32
img_height = 400
img_width = 400

# Define training transforms
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomResize(256, 400),
    transform.RandomRotation(30),
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

# Load split datasets (created in external splitting script)
train_dir = "./EMODB Database/RGB_IMG_Split/train/"
test_dir = "./EMODB Database/RGB_IMG_Split/test/"
# Reload the newly created training and testing datasets, applying transforms
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# --------------------- 2. Construct Model - ResNet50 ---------------------
model_rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
for name, param in model_rn.named_parameters():
    if 'layer1' in name or 'layer2' in name:
        param.requires_grad = False
num_features = model_rn.fc.in_features

# Use sequential to add regularization layers
model_rn.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Flatten(),
    nn.Linear(num_features, 4096),
    nn.BatchNorm1d(4096),
    nn.Dropout(p=0.55),
    nn.Linear(4096, 1024),
    nn.BatchNorm1d(1024),
    nn.Linear(1024, len(train_dataset.classes)),
    nn.Softmax(dim=1)
)

# --------------------- 3. Train & Evaluate Model ---------------------
# a.) Set device (Cuda or CPU) %%%%%%%%%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')
model_rn = model_rn.to(device)

n_epochs = 15
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
optimizer = optim.Adam(model_rn.parameters(), lr=1e-4, weight_decay=1e-1)
# optimizer = optim.SGD()
# Decay LR by a factor of 0.1 every [step_size] epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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
    # scheduler.step()
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


# Load train and test data using dataloader
train_loader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
)
test_loader = data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
)

# f.) Execute model training and testing %%%%%%%%%%
train_loss = []
test_loss = []
train_acc = []
test_acc = []
print('-' * 10)
since = time.time()
for epoch_i in range(n_epochs):
    # since_epoch = time.time()
    print(f'Epoch {epoch_i + 1}/{n_epochs}')
    # TRAINING + Display epoch stats
    train_model(model_rn, criterion, optimizer, exp_lr_scheduler)
    # TESTING + Display epoch stats
    test_model(model_rn)
    print('-' * 10)
# Print total time of training + testing
time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print('-' * 10)

# Print stats of each fold to console:
timestamp = datetime.datetime.now().strftime("%d-%m__%H-%M")
print(f'Simulation complete {timestamp}.')
print('PILOT RESULTS:')
print(f'Last accuracy on epoch {n_epochs}:')
print(test_acc[-1])
print(f'\nBest accuracy over {n_epochs} epochs:')
print(max(test_acc))
print('-' * 10)

# PLOT TRAIN AND TEST RESULTS
plt.figure()
plt.suptitle('EMODB DCNN Pilot Results')
# g.i.) Plot the train and test loss (exp dec) %%%%%%%%%%
plt.subplot(1, 2, 1)
plt.plot(list(range(1, (n_epochs)+1)), train_loss, 'b', label='Train Loss')
plt.plot(list(range(1, (n_epochs)+1)), test_loss, 'r', label='Test Loss')
plt.title('Simulation Loss')
plt.legend(loc='upper right')
plt.xlabel('N epochs')
plt.ylabel('Average loss')
plt.grid()

# g.ii.) Plot the train and test acc (exp inc) %%%%%%%%%%
plt.subplot(1, 2, 2)
# Plot lines to show start and end of each fold
plt.plot(list(range(1, (n_epochs)+1)), train_acc, 'b', label='Train Acc')
plt.plot(list(range(1, (n_epochs)+1)), test_acc, 'r', label='Test Acc')
ax = plt.gca()      # get current axis
ax.set_ylim([0, 102])
plt.title('Prediction Accuracy')
plt.legend(loc='lower right')
plt.xlabel('N epochs')
plt.ylabel('Model accuracy (%)')
plt.grid()

plt.tight_layout()
save_path = './EMODB_Pilot_Results/'
plt.savefig(f'{save_path}EMODB_PilotResults_{timestamp}.png')
plt.show()

# --------------------- 4. Save & Load Params ---------------------
# a.) Save trained model parameters %%%%%%%%%%
# filename = f"params_{timestamp}.pt"
# torch.save(model_rn.state_dict(), f'rn50SavedParams/EMODB/{filename}')

