#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Train a network
        Args:
            dataloaders (dict) : dictionary with dataloaders
            dataset_sizes (dict) : dictionary of len of dataloaders
            model : model to be trained
            criterion : criterion
            optimizer : optimizer
            save path : file path to save the model to
            scheduler: scheduler
            device (str): 'cuda' or 'cpu' 
            num_epochs : num epochs to train for, (default = 25)
            plot (bool) : whether to plot or not

        Returns:
            model : trained model
'''
import time
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import copy 
import numpy as np
import matplotlib.pyplot as plt

def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, save_path, scheduler, device='cuda', num_epochs=25, plot=False):
    model.to(device)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # track our loss history over the epochs for plotting
    loss_history = []
    counter = []
    count = 0 

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs ))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (data, target) in enumerate(dataloaders[phase]):
                inputs = data.to(device)
                labels = target.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.to(device))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.mean((preds == labels.data).type(torch.FloatTensor))
                
                if plot and batch_idx % 10 == 0:
                    count += 10
                    counter.append(count)
                    loss_history.append(loss.item()*inputs.size(0))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    if plot:
        plt.plot(counter,loss_history)
        plt.show()
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
