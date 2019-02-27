#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import *
import numpy as np
from glob import glob
import torch.nn as nn
from PIL import ImageFile
from datasets_and_dataloaders import *
from utils import *
from model_scratch import Net
from train import *
from predict import *

# check if cuda is available
use_cuda = torch.cuda.is_available()

# get array of images 
human_files = np.array(glob("humanImages/*/*"))
my_photos = np.array(glob("*.jpg"))
dog_files = np.array(glob("dogImages/*/*/*"))
dog_classes = np.array(glob("dogImages/train/*"))
num_dog_classes = len(dog_classes)
# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
print('There are %d total personal images.' % len(my_photos))
print('There are %d total dog classes.' % num_dog_classes)

## Specify appropriate transforms, and batch_sizes
train_path = "dogImages/train/"
valid_path = "dogImages/valid/"
test_path = "dogImages/test/"
batch_size = 32

# transforms
train_transforms = transforms.Compose([
    transforms.CenterCrop(350),
    transforms.RandomAffine(4, translate=(0.2,0.2), scale=None, shear=0.2, resample=False, fillcolor=0),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]) 
])
transforms_ = {'train':train_transforms, 'test':test_transforms}
directories = {'train':train_path, 'valid':valid_path, 'test':test_path}

# datasets
datasets_ = create_datasets(directories, transforms_, valid=True)

# dataloaders
dataloaders = create_dataloaders(datasets_, batch_size, shuffle=True)
train_dataloader = dataloaders['train']
valid_dataloader = dataloaders['valid']
test_dataloader = dataloaders['test']
dataset_sizes = {"train":len(train_dataloader), "valid":len(valid_dataloader), "test":len(test_dataloader)}
print(dataset_sizes)

# instantiate the models
model_scratch = Net(len(datasets_['train'].classes))
model_transfer = models.vgg16(pretrained=True)

# freeze parameters so we don't backprop through them
for param in model_transfer.parameters():
    param.requires_grad = False

# replace classifier    
in_features = model_transfer.classifier[6].in_features 
out_features = len(datasets_['train'].classes)
model_transfer.classifier[6] = nn.Sequential(nn.Linear(in_features, out_features))

# move tensors to GPU if CUDA is available
if use_cuda:
  device = 'cuda'
  model_scratch.cuda()
  model_transfer.cuda()

# select loss function
criterion = nn.CrossEntropyLoss()

# select optimizer
optimizers_ = {"scratch":optim.SGD(model_scratch.parameters(), lr=0.01, momentum=0.7), 
				"transfer":optim.Adam(model_transfer.classifier[6].parameters(), lr = 0.00003)}

# select scheduler 
schedulers_ = {"scratch":lr_scheduler.StepLR(optimizers_["scratch"],step_size=25,gamma=0.1,last_epoch=-1),
				"transfer": lr_scheduler.StepLR(optimizers_["transfer"],step_size=20,gamma=0.1,last_epoch=-1)}

# train the model from scratch
# model_scratch = train_model(dataloaders, dataset_sizes, model_scratch, criterion, optimizer 
                      # , save_path='model_scratch.pt', scheduler=scheduler, device=device, num_epochs=30, plot=True)
# call test function 
ImageFile.LOAD_TRUNCATED_IMAGES = True
test(dataloaders, model_scratch, criterion, use_cuda)

# train the model from transfer learning
# model_transfer = train_model(dataloaders, dataset_sizes, model_transfer, criterion, optimizer 
                      # , save_path='model_transfer-vgg16-v1.pt', scheduler=scheduler, device=device, num_epochs=30, plot=True)

# call test function 
test(dataloaders, model_transfer, criterion, use_cuda)