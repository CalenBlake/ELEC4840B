"""
# -----------------------------------
# Load a trained model and quantify performance on a test set.
# Create confusion matrix plot output to be included on poster.
# Generic build to allow for testing of models trained on either dataset.
#
# Author: Calen Blake
# Date: 11-06-23
# NOTE: 
# -----------------------------------
"""
# --------------------- Import necessary libraries ---------------------
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import time
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from sklearn.utils.multiclass import type_of_target

# --------------------- 1. Define functions ---------------------
def get_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct/total
    return accuracy

# --------------------- 2. Load data & model ---------------------
# Load test dataset
batch_size = 32
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
data_dir = "./EMODB Database/RGB_IMG_Split/test/"
test_dataset = datasets.ImageFolder(data_dir, transform=test_transforms)
test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False, pin_memory=True
    )

# Load model
model = torch.load('EMODB-bestModel.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')
model.to(device)
model.eval()

# --------------------- 3. Pass test set ---------------------
predictions = []
true_classes = []

# Forward pass of images through trained model
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        predictions.append(preds.cpu().numpy())
        true_classes.append(labels.cpu().numpy())

accuracy = get_accuracy(outputs, labels)
print(f'The test accuracy is: {accuracy*100:.2f}%')

# --------------------- 4. Confusion matrix output ---------------------
# ensure predictions and true_classes in correct format
y_pred = np.concatenate(
    (predictions[0], predictions[1],
    predictions[2], predictions[3]), axis=None
    )
y_true = np.concatenate(
    (true_classes[0], true_classes[1],
    true_classes[2], true_classes[3]), axis=None
    )
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Change the name of class labels to english variants
# classes = []

# save confusion matrix figure as SVG format!
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot(cmap=plt.cm.Blues)
plt.title("EMODB Test Classification")
plt.show()






