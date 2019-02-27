#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from PIL import Image
import torchvision
from utils import imshow, face_detector, dog_detector


def load_image(img_path):
	""" 
	Load the image and return the predicted breed.
	Args:
		img_path (str): file path for the image
	Returns:
		img (tensor): img as a tensor to be served to the model
	"""
	img = Image.open(img_path)
	trans1 = torchvision.transforms.Resize((224, 224))
	trans2 = torchvision.transforms.ToTensor()
	img = trans2(trans1(img))
	img.unsqueeze_(0)
	return img

def predict_breed_transfer(img, model):
	"""
	Predict the breed of a dog from an image.
	Args:
		img (tensor): img as a tensor to be served to the model
		model: trained model
	Returns:
		pred (dict): dictionary containing the probability and predicted class
		img (tensor): img as a tensor
	"""
	model.eval().cpu()
	outputs = model(img)
	outputs = torch.exp(outputs)
	probs, class_indices = outputs.topk(5)
	probs = list(probs.cpu().detach().numpy()[0])
	class_indices = list(class_indices.numpy()[0])
	# pred contains the predicted topk probabilities, and class indices
	pred = dict(zip(probs, class_indices))
	# return single label, prediction, image
	return  pred, img.cpu().squeeze(0)


def run_app(model, img_path, labels, dog_images, add_file_path=None):
	""" 
	Take in an image and predict the dog breed for a detected dog, or human face.
	Args:
		model: trained model to predict the dog breed
		img_path (str): img path for the image
		labels (list): list of dog classes
		dog_images (df): dataframe containing one image per dog breed
		add_file_path (str): filepath to be prepended to the haarcascade xml file (if using colab)
	"""
	faces = face_detector(img_path, add_file_path+"haarcascade_frontalface_alt.xml")
	is_dog = dog_detector(img_path)
	if faces or is_dog:
	    img = load_image(img_path)
	    pred, img = predict_breed_transfer(img, model)
	    # get the top prediction
	    max_prob = max(pred)
	    # get the names of the predicted and actual class
	    pred_class = int(pred[max_prob])

	    if is_dog: 
	        name= "dog! You must be a"
	            # set the title of the plot
	    else:
	        name= "human. You most resemble a"
	    
	    title = "Hello {} {}".format(name,labels[pred_class].split('.')[1])
	    dog_img = dog_images[dog_images.index == pred_class]['image_fps'].values[0]
	    dog_img= load_image(dog_img)
	    dog_img.squeeze_()
	    images = [img, dog_img]
	    imshow(torchvision.utils.make_grid(images), norm=False,title=title)
	    
	else:
	    print("No faces detected.")
    
    

