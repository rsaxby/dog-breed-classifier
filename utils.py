#!/usr/bin/env python
# -*- coding: utf-8 -*-


from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''

    # define VGG16 model
    VGG16 = models.vgg16(pretrained=True)

    # check if CUDA is available
    use_cuda = torch.cuda.is_available()

    # move model to GPU if CUDA is available
    if use_cuda:
    	VGG16 = VGG16.cuda()

    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    img = Image.open(img_path)
    
    # transforms for  img
    data_transforms = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
      ])
    # apply transforms and move to gpu
    img = data_transforms(img).cuda()

    # add a dimension for the batch size
    img = img.unsqueeze(0)
    # feed img into pretrained model
    output = VGG16(img)
    
    # get the prediction
    _, pred_t = torch.max(output, 1)
    # convert from tensor to np array
    pred = torch.squeeze(pred_t).cpu().numpy()
    
    return pred # predicted class index


def face_detector(img_path, haarcascade_frontalface_alt_path="haarcascade_frontalface_alt.xml"):
	"""
	Returns "True" if face is detected in image stored at img_path
	Args:
        img_path (str): path to an image
        haarcascade_frontalface_alt_path (str): path to the haarcascade xml file
	Returns:
		bool : "True" if a face is detected in the image
	"""
	# extract pre-trained face detector
	face_cascade = cv2.CascadeClassifier(haarcascade_frontalface_alt_path)
	img = cv2.imread(img_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray)
	return len(faces) > 0

def dog_detector(img_path):
	"""
	Returns "True" if a dog is detected in the image stored at img_path
	Args:
		img_path: path to an image
	Returns:
		bool : "True" if a dog is detected in the image
	"""
	pred = VGG16_predict(img_path)
	return (pred >= 151 and pred <= 268) # true/false

def create_dog_image_df(dataset, dog_files):
	""" 
	Create a dataframe with one dog image per breed
	Args:
		dataset: torch dataset (train or valid)
		dog_files (np array): dog images
	Returns:
		dog_images (pandas df): df which contains one image and the file paths for each breed
	"""
	dog_images = pd.DataFrame(data=None, index=[x.split('.')[0] for x in dataset.classes])
	dog_images['classes'] = dataset.classes
	dog_img_paths = [x.split('/',9)[9:][0] for x in dog_files]
	one_img_per_class_imgs = []
	one_img_per_class_fps = []
	for c in dog_images['classes']:
	  one_img_per_class_imgs.append(next(x[1] for x in dog_img_paths if str(c) in x))
	  one_img_per_class_fps.append(next(x for x in dog_files if str(c) in x))
	dog_images['images'] = one_img_per_class_imgs
	dog_images['image_fps'] = one_img_per_class_fps
	dog_images.index = np.arange(133)
	return dog_images



def imshow(image, norm=True,ax=None, title=None):

	"""
	Imshow for Tensor
	Args:
		image (tensor)
		norm (bool) : whether image was normalized (if so, we need to undo preprocessing)
		title (str) : the title for the plot
	"""
	if ax is None :
		fig, ax = plt.subplots(figsize=(4, 5))

	# PyTorch tensors assume the color channel is the first dimension
	# but matplotlib assumes is the third dimension
	image = image.numpy().transpose((1, 2, 0))
	# Undo preprocessing
	if norm:
	    mean = np.array([0.485, 0.456, 0.406])
	    std = np.array([0.229, 0.224, 0.225])
	    image = std * image + mean

	# Image needs to be clipped between 0 and 1 or it looks like noise when displayed
	image = np.clip(image, 0, 1)
	ax.set_title("{}".format(title))
	ax.grid(False)
	plt.axis('off')
	ax.imshow(image)
    
	return ax

def test(dataloaders, model, criterion, use_cuda):
	""" 
	Test the trained model.
	Args:
		loaders (dict): dictionary of dataloaders
		model: trained model
		criterion: criterion
		use_cuda (bool): True/False - whether to use gpu
	"""
	# monitor test loss and accuracy
	test_loss = 0.
	correct = 0.
	total = 0.

	model.eval()
	for batch_idx, (data, target) in enumerate(dataloaders['test']):
	    # move to GPU
	    if use_cuda:
	        data, target = data.cuda(), target.cuda()
	    # forward pass
	    output = model(data)
	    # calculate the loss
	    loss = criterion(output, target)
	    # update average test loss 
	    test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
	    # convert output probabilities to predicted class
	    pred = output.data.max(1, keepdim=True)[1]
	    # compare predictions to true label
	    correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
	    total += data.size(0)
	        
	print('Test Loss: {:.6f}\n'.format(test_loss))

	print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
	    100. * correct / total, correct, total))

