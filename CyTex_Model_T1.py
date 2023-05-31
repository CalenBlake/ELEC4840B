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
from sklearn.model_selection import StratifiedKFold

# --------------------- 1. Load & Preprocess Data ---------------------
# a.) load data %%%%%%%%%%
batch_size = 32
img_height = 400
img_width = 400

# Plot to separate window and remove Matplotlib version 3.6 warnings
# matplotlib.use("TkAgg")

# Define training transforms
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Resize(256, antialias=False),
    # transforms.RandomRotation(15),
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

# Load whole dataset
data_dir = "./EMODB Database/RGB_IMG_noOverlap/"
# Reload the newly created training and testing datasets, applying transforms
train_dataset_imf = datasets.ImageFolder(data_dir, transform=train_transforms)
test_dataset_imf = datasets.ImageFolder(data_dir, transform=test_transforms)

# e.) plot sample of transformed train batch
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
# OLD SYNTAX: model_rn = models.resnet50(pretrained=True)
model_rn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# model_rn = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
# FREEZE Weights of the first two blocks, retrain remaining by default
# named_parameters() returns (str, Parameter) – Tuple containing the name and parameter
# Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
for name, param in model_rn.named_parameters():
    if 'layer1' in name or 'layer2' in name:
        # Uncomment to see names of layers and contents
        # print(name)
        param.requires_grad = False

# Alternative weight freezing method highlighted in the following link:
# https://discuss.pytorch.org/t/model-named-parameters-will-lose-some-layer-modules/14588

# get number of input features for the last fully connected layer
# This will feed into the following layers we create at the output of the ResNet model
num_features = model_rn.fc.in_features
print('\n--------------------')
print('input features of final resnet layer (Pre CyTex additions):')
print(num_features)
print('--------------------')
# num_features = 2048

# View the architecture of the ResNet50 model
# print('\n--------------------')
# print('Pre-modification model architecture:')
# print(model_rn)
# print('--------------------')

# Check shape of example data
# print('\n--------------------')
# examples = enumerate(train_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# print('The shape of the example data is:')
# print(example_data.shape)
# # returns: torch.Size([32, 3, 400, 400])
# print('--------------------')

# Use sequential to add regularization layers
model_rn.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Flatten(),
    nn.Linear(num_features, 4096),
    nn.BatchNorm1d(4096),
    nn.Dropout(p=0.55),
    nn.Linear(4096, 1024),
    nn.BatchNorm1d(1024),
    # ASK ALI: Should modify shape to be num of classes!?
    nn.Linear(1024, len(train_dataset_imf.classes)),
    # dim=1 applies the softmax operation over the first dimension of the tensor
    # Returns probability distribution over the classes for each example in each batch
    nn.Softmax(dim=1)
)

# View new model architecture
# print('\n--------------------')
# print('New modified model architecture:')
# print(model_rn)
# print('--------------------')



# Pass test tensor through model to check shape
# output_tensor = model_rn(example_data)
# print('\n--------------------')
# print('The shape of the example data, after passing through the model, is:')
# print(output_tensor.shape)
# # returns: torch.Size([32, 7]) = [batch_size, num_classes]
# print('--------------------')


# --------------------- 3. Train & Evaluate Model ---------------------
# Initially train for minimal epochs and check results then scale up to ~200 epochs
# a.) Set device (Cuda or CPU) %%%%%%%%%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')
model_rn = model_rn.to(device)

n_epochs = 15
n_batches = np.ceil(len(train_dataset_imf)/batch_size)

# b.) Print some useful info before training
print('\n--------------------')
print('Total data samples: ', len(train_dataset_imf) + len(test_dataset_imf))
print('Train data samples: ', len(train_dataset_imf))
print('Test data samples: ', len(test_dataset_imf))
print(f'batch size: {batch_size:.0f} --> training batches: {n_batches:.0f}')
print(f'epochs: {n_epochs:.0f} --> total batches: {(n_epochs*n_batches):.0f}')
print('--------------------')

# d.) Create callable functions for model training & testing %%%%%%%%%%
def train_model(model, criterion, optimizer):
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
    # calculate + print: loss and acc over epoch_i
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100 * running_corrects / len(train_dataset)
    print(f'train loss: {epoch_loss}, train acc: {epoch_acc}')
    # append epoch results to corresponding lists
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    total_train_loss.append(epoch_loss)
    total_train_acc.append(epoch_acc)


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
        total_test_loss.append(epoch_loss)
        total_test_acc.append(epoch_acc)

# e.) Employ stratified k-fold splitting and loop
k = 5
# set list of labels/targets and dummy var x
# ***CHECK: y should be identical for train and test datasets due to the same loading procedure and organisation
y = train_dataset_imf.targets
x = np.zeros(len(y))
# alter random_state to alter results or make reproducible
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
# setup k-fold stats
kf_last_acc = []
kf_last_loss = []
kf_best_acc = []
kf_best_loss = []
# setup total stats (spanning all folds)
total_train_acc = []
total_train_loss = []
total_test_acc = []
total_test_loss = []

# ========== Main k-fold Loop ==========
print('\nINITIATING MODEL TRAINING & TESTING...')
for fold, (train_indices, test_indices) in enumerate(skf.split(x, y)):
    print(f"Training on fold {fold + 1}/{k}")
    # ========== Define Model for each k-fold ==========
    # reinitialize the model parameters for each new fold
    model_rn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze weights of first two layers
    for name, param in model_rn.named_parameters():
        if 'layer1' in name or 'layer2' in name:
            param.requires_grad = False
    num_features = model_rn.fc.in_features
    # regularization and output layers
    model_rn.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Flatten(),
    nn.Linear(num_features, 4096),
    nn.BatchNorm1d(4096),
    nn.Dropout(p=0.55),
    nn.Linear(4096, 1024),
    nn.BatchNorm1d(1024),
    nn.Linear(1024, len(train_dataset_imf.classes)),
    nn.Softmax(dim=1)
    )
    # Define optimization criteria
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_rn.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # Send model to cuda or other device
    model_rn = model_rn.to(device)
    # ========== END: Define Model for each k-fold ==========
    # Reset epoch statistics for each new fold
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    train_dataset = data.Subset(train_dataset_imf, train_indices)
    test_dataset = data.Subset(test_dataset_imf, test_indices)
    # Load train and test data using dataloader
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False, pin_memory=True
    )

# f.) Execute model training and testing %%%%%%%%%%
    print('-' * 10)
    since = time.time()
    for epoch_i in range(n_epochs):
        print(f'Epoch {epoch_i + 1}/{n_epochs}')
        # TRAINING + Display epoch stats
        train_model(model_rn, criterion, optimizer)
        # TESTING + Display epoch stats
        test_model(model_rn)
        # step the scheduler on an epoch passing basis!
        # scheduler.step()
        print('-' * 10)
    # Print total time of training + testing
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print('-' * 10)
    # Store performance metrics for each of the k-folds
    # Use negative indexing to access last item of list, correlating to test stat from epoch n/n
    kf_last_acc.append(test_acc[-1])
    kf_last_loss.append(test_loss[-1])
    kf_best_acc.append(max(test_acc))
    kf_best_loss.append(min(test_loss))

# Print stats of each fold to console:
timestamp = datetime.datetime.now().strftime("%d-%m__%H-%M")
print(f'Simulation complete {timestamp}.')
print('K-FOLD RESULTS:')
print(f'Accuracy on epoch {n_epochs}/{n_epochs}:')
for i in range(k):
    print(f'fold {i+1} test accuracy: {kf_last_acc[i]:.2f}%')
print(f'\nBest accuracy over {n_epochs} epochs:')
for i in range(k):
    print(f'fold {i+1} test accuracy: {kf_best_acc[i]:.2f}%')
print(f'\nAverage best accuracy across {k} folds: {np.average(kf_best_acc):.2f}%')
print('-' * 10)

# PLOT TRAIN AND TEST RESULTS
plt.figure()
plt.suptitle('EMODB DCNN Training Results')
# g.i.) Plot the train and test loss (exp dec) %%%%%%%%%%
plt.subplot(1, 2, 1)
# Plot lines to show start and end of each fold
plt.axvline(x=1,color='yellow',label='k-folds')
for i in range(1, (k*n_epochs)+1):
    if (i%n_epochs == 0):
        plt.axvline(x=i,color='yellow')
plt.plot(list(range(1, (k*n_epochs)+1)), total_train_loss, 'b', label='Train Loss')
plt.plot(list(range(1, (k*n_epochs)+1)), total_test_loss, 'r', label='Test Loss')
plt.title('Simulation Loss')
plt.legend(loc='upper right')
plt.xlabel('N epochs')
plt.ylabel('Average loss')
plt.grid()

# g.ii.) Plot the train and test acc (exp inc) %%%%%%%%%%
plt.subplot(1, 2, 2)
# Plot lines to show start and end of each fold
plt.axvline(x=1,ymin=0,ymax=100,color='yellow',label='k-folds')
for i in range(1, (k*n_epochs)+1):
    if (i%n_epochs == 0):
        plt.axvline(x=i,color='yellow')
plt.plot(list(range(1, (k*n_epochs)+1)), total_train_acc, 'b', label='Train Acc')
plt.plot(list(range(1, (k*n_epochs)+1)), total_test_acc, 'r', label='Test Acc')
ax = plt.gca()      # get current axis
ax.set_ylim([0, 102])
plt.title('Prediction Accuracy')
plt.legend(loc='lower right')
plt.xlabel('N epochs')
plt.ylabel('Model accuracy (%)')
plt.grid()

plt.tight_layout()
plt.savefig(f'EMODB_TrainResults_{timestamp}.png')
plt.show()

# --------------------- 4. Save & Load Params ---------------------
# a.) Save trained model parameters %%%%%%%%%%
# filename = f"params_{timestamp}.pt"
# torch.save(model_rn.state_dict(), f'rn50SavedParams/EMODB/{filename}')
"""
# b.) Load trained model parameters %%%%%%%%%%
model = model_rn
load_file = f'rn50 saved params - EMODB/model_params_12-04__15-48.pt'
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
"""
