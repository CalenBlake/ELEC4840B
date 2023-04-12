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
from torchvision.utils import save_image
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
import time
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
# t_images, t_labels = next(iter(train_loader))
# # Plot a sample of 6 images from the batch
# fig, axs = plt.subplots(2, 3, figsize=(12, 6))
# for i, ax in enumerate(axs.flatten()):
#     ax.imshow(t_images[i].permute(1, 2, 0))
#     class_name = train_dataset.classes[t_labels[i]]
#     ax.set_title(f"Class: {class_name}")
# plt.suptitle('Random Sample of Training Data')
# plt.tight_layout()
# plt.show()

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
# print('\n--------------------')
# print('Pre-modification model architecture:')
# print(model_rn50)
# print('--------------------')

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
# print('\n--------------------')
# print('New modified model architecture:')
# print(model_rn50)
# print('--------------------')



# Pass test tensor through model to check shape
output_tensor = model_rn50(example_data)
print('\n--------------------')
print('The shape of the example data, after passing through the model, is:')
print(output_tensor.shape)
# returns: torch.Size([32, 7]) = [batch_size, num_classes]
print('--------------------')


# --------------------- 3. Train & Evaluate Model ---------------------
# Initially train for minimal epochs and check results then scale up to ~200 epochs
# a.) Set device (Cuda or CPU) %%%%%%%%%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_rn50 = model_rn50.to(device)

n_epochs = 50
n_batches = np.ceil(len(train_dataset)/batch_size)

# b.) Print some useful info before training
print('\n--------------------')
print('Total data samples: 1211')
print('Train data samples: ', len(train_dataset))
print('Test data samples: ', len(test_dataset))
print(f'batch size: {batch_size:.0f} --> training batches: {n_batches:.0f}')
print(f'epochs: {n_epochs:.0f} --> total batches: {(n_epochs*n_batches):.0f}')
print('--------------------')

# c.) Define loss function, optimizer, lr scheduler and run-time stats %%%%%%%%%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_rn50.parameters(), lr=0.1, momentum=0.1)
# Decay LR by a factor of 0.1 every [step_size] epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
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
        # step the scheduler
        scheduler.step()
        # calculate running loss and acc
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()
        # FORWARD END ----------
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
    # print time per epoch for train and test cumulative pass
    # t_elapsed_epoch = time.time() - since_epoch
    # print(f'Training complete in {t_elapsed_epoch // 60:.0f}m {t_elapsed_epoch % 60:.0f}s')
    # print('-' * 10)
# Print total time of training + testing
time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print('-' * 10)
print()

# f.i.) Plot the train and test loss (exp dec) %%%%%%%%%%
plt.figure()
plt.plot(list(range(1, n_epochs+1)), train_loss, 'b')
plt.plot(list(range(1, n_epochs+1)), test_loss, 'r')
plt.title('Simulation Loss')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('N epochs')
plt.ylabel('Average loss')
plt.grid()
plt.tight_layout()
plt.show()

# f.ii.) Plot the train and test acc (exp inc) %%%%%%%%%%
plt.figure()
plt.plot(list(range(1, n_epochs+1)), train_acc, 'b')
plt.plot(list(range(1, n_epochs+1)), test_acc, 'r')
plt.title('Prediction Accuracy')
plt.legend(['Train Acc', 'Test Acc'], loc='upper right')
plt.xlabel('N epochs')
plt.ylabel('Model accuracy (%)')
plt.grid()
plt.tight_layout()
plt.show()

# --------------------- 4. Save & Load Params Visualize Results ---------------------
# b.) Save trained model parameters %%%%%%%%%%
timestamp = datetime.datetime.now().strftime("%d-%m__%H-%M")
filename = f"model_params_{timestamp}.pt"
torch.save(model_rn50.state_dict(), f'RN50 - Saved Params/{filename}')

# c.) Load trained model parameters %%%%%%%%%%
model = model_rn50
load_file = f'RN50 - Saved Params/model_params_12-04__15-48.pt'
model.load_state_dict(torch.load(load_file))

# --------------------- 5. Test Model & Visualize Results ---------------------
predictions = []
true_classes = []
# a.) Pass test data through the trained model %%%%%%%%%%
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predictions.append(preds.cpu().numpy())
        true_classes.append(labels.cpu().numpy())

predictions = np.concatenate(predictions)
true_classes = np.concatenate(true_classes)

# b.) Plot confusion matrix to visualize acc on each class %%%%%%%%%%
# cm = confusion_matrix(true_classes, predictions, labels=test_dataset.classes)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
# disp.plot()
# plt.show()