#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torchvision import datasets

def create_datasets(directories, transforms, valid=False, split_amount=None):
	'''
	Create a dataset with ImageFolder
	        Args:
	        	directories (dict) : dictionary with train and test directories
				transforms (dict) : dictionary with train and test transforms
				split (bool) : True if a further split of the train set is needed (for valid), default is False
	            split_amount (float): percent for split, default is None

	        Returns:
	        	train, valid (torch datasets)
	'''
	# directories
	train_dir = directories['train']
	test_dir = directories['test']
	# transforms
	train_transforms = transforms['train']
	test_transforms = transforms['test']
	# create datasets
	train = datasets.ImageFolder(train_dir, transform=train_transforms)
	test = datasets.ImageFolder(test_dir, transform=test_transforms)

	if valid:
		if split_amount:
			total = len(train)
			split = int(total*split_amount)
			torch.manual_seed(999)
			train, valid = torch.utils.data.random_split(train, (split, total-split))
		else:
			valid = datasets.ImageFolder(directories['valid'], transform=test_transforms)
		return {"train":train, "valid":valid, "test":test}
	else:
		return {"train":train, "test":test}


def create_dataloaders(datasets, batch_size, shuffle=True):
	'''
	Create a dataloaders from torch datasets
	        Args:
	        	datasets (dict) : dictionary with datasets
	        Returns:
	        	dataloaders (dict) : dictionary with dataloaders
	'''
	train_dataloader = torch.utils.data.DataLoader(datasets["train"], batch_size=batch_size, shuffle=shuffle)
	test_dataloader = torch.utils.data.DataLoader(datasets["test"], batch_size=batch_size, shuffle=shuffle)
	if "valid" in datasets:
		valid_dataloader = torch.utils.data.DataLoader(datasets["valid"],  batch_size=batch_size, shuffle=shuffle)
		return {"train": train_dataloader, "valid": valid_dataloader, "test": test_dataloader}
	else:
		return {"train": train_dataloader, "test": test_dataloader}


