"""
# -----------------------------------
# Tutorial exercise from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# This exercise details training a CNN for image classification using transfer learning.
# The model will be used to classify images of ants and bees.
#
# Author: Calen Blake
# Date: 20-03-23
# NOTE:
# -----------------------------------
"""
from __future__ import print_function, division
if __name__ == '__main__':
    # --------------------- Import necessary libraries ---------------------

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    import torch.backends.cudnn as cudnn
    import numpy as np
    import torchvision
    from torchvision import datasets, models, transforms
    import matplotlib.pyplot as plt
    import time
    import os
    import copy
    # Added by CALEN: To see ResNet layers
    from torchsummary import summary

    # Enable CUDA Deep Neural Network to benchmark multiple convolution algorithms and use the fastest
    cudnn.benchmark = True
    # Enable interactive mode
    plt.ion()

    # --------------------- 1. Load Data ---------------------
    # Data augmentation and normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # ???: how to find mean and st-dev for normalization
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Just normalization for validation
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Set data directory to ant and bee image data
    data_dir = 'transferTut_data'
    # Set image folder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    # Load the data in the declared folder
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    # Measure the amount of images in the training set and validation set
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # Creates a list of class names read from the training set ('ants', 'bees')
    class_names = image_datasets['train'].classes

    # Enable cuda (if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Visualize images fom the training set to understand the data augmentations made
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


    # Get a batch of training data => CAUSED PROBLEMS WITHOUT IF NAME == MAIN HOTFIX
    inputs, classes = next(iter(dataloaders['train']))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    # Plot the grid of images, ouput is 4 of the training images stacked horizontally.
    imshow(out, title=[class_names[x] for x in classes])

    # --------------------- 2. Train Model ---------------------
    # NOTE: We define our own function which will train a model on the dataset
    # and then train a separate model and compare the results of the two.
    # It will continue iteratively doing so. Eventually the optimal model will be used.
    # This step will likely not be necessary in my research as we have settled on using
    # ResNet and other various lightweight models.
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    # --------------------- 2.1. Visualize Model Predictions ---------------------
    # Here we create a function to display predictions of a small set of images
    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images //2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {class_names[preds[j]]}')
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    # --------------------- 3. Fine-tuning the Conv-net ---------------------
    # Here we load a pretrained model and reset the final fully connected layer
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # --------------------- 3.1. Train and evaluate the Conv-net ---------------------
    # Fit the model and print evaluation statistics
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)
    # Plot a few images with their class prediction
    # NOTE: For some reason plots 8 figures in separate windows ???
    visualize_model(model_ft)

    # --------------------- 4. Conv-net as a Feature Extractor ---------------------
    # Here we FREEZE the network apart from the final layer.
    # This means that the weights on the convolutional layers will not need to be retrained
    # (at least other than the final layer). Thus, it is the final layer which will customise
    # The general model to our specific dataset.
    # 'requires_grad = False' will freeze the params so grads aren't computed in backward().
    # read more: https://pytorch.org/docs/master/notes/autograd.html
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Can print model to view the last layer - UNCOMMENT FOLLOWING LINE
    # print(model_conv)
    # output is: (fc): Linear(in_features=512, out_features=1000, bias=True)
    # Alternatively for more detail - UNCOMMENT FOLLOWING LINE INSTEAD
    # summary(model_conv, (3, 224, 224))



    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    # Here we reinitialize the last fully connected layer to have 2 output features
    # and by default the new layer will retrain (requires_grad=True)
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    # --------------------- 4.1. Train and evaluate the Conv-net ---------------------
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=25)
    visualize_model(model_conv)

    plt.ioff()
    plt.show()


