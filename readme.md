##Dog Breed Classifier

This was the second project in the Udacity Deep Learning Nanodegree program. This was the CNN unit capstone project, where we trained two models (one from scratch, and one using transfer learning) to classify a dog breed from any user-supplied image as input. 

If a dog is detected in the image, it provides an estimate of the dog's breed. If a human is detected (using OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)), it provides an estimate of the dog breed that the human is most resembling.

### Data

* 13233 total human images
* 8351 total dog images
  * 133 total dog classes

### Data Preprocessing

I chose the following transforms:

```python
transforms.RandomAffine(4, translate=(0.2,0.2), scale=None, shear=10, resample=False, fillcolor=0),
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
transforms.RandomRotation(10),
transforms.Resize((224, 224)),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
transforms.ToTensor()
```

RandomAffine to translate and shear the images to account for possible slight differences in the photo angles and positions of the dogs within the photos. 

ColorJitter to account for possible differences in lighting, contrast, and saturation within the photos. 

RandomRotation again to add variability to the photos so the network is able to generalize and adapt to various positions and angles of the dogs within the photos.

I resized to 224 and normalized using the following values because that's what a pretrained (ImageNet) network would require, and I decided to adhere to it for my model from scratch as well for consistency.

Finally, I convereted the array to a tensor to be fed into the network. 

### Architecture: Model from scratch

My initial model architecture was just an implementation of VGG16, however I found that I was severely overfitting due to the depth of the network relative to the small dataset.

The final architecture I settled on is a very pruned down version of VGG16, in which I significantly reduced the number of filters as well as the number of layers. This resulted in much better accuracy, and much less of a gap between the statistics (loss and accuracy) between my train/validation sets, and my test set.

I originally had 3 fully connected layers and reducing it to 2 also helped reduce my overfitting. I included a dropout, which I originally set to 30%, but increasing it to 50% also helped increase generability. I used relu activation because it's standard to use in CNN's for its efficiency/fast compute time relative to other activation functions like tanh or sigmoid. 

My model from scratch achieved 15% accuracy on the test set. If I were to try to improve this model's accuracy further, I would next try batch normalization and leaky relu.

### Transfer Learning : VGG16

I used vgg16 as my base network because it's a relatively shallow network making it faster to train, and it has been trained on ImageNet images (including the dog breeds in question). I froze all layers, replacing and training only the classifier. For the classifier, I opted for a single fc layer to avoid overfitting. With this configuration, I was able to achieve 85% accuracy.

### Final Thoughts

The model's dog breed predictions (on dog images) are mostly accurate, however it has trouble distinguishing dog breeds when both are the same color. Or, if a normally distinguishing feature (like perked ears on a Boston Terrier) are not visible in the picture. Some areas of improvement include distinguishing between breeds when they share features with another breed (like markings or color and hair type). I'd also like to create an app of this project, and add more features to it, like detecting multiple faces and multiple dogs in a picture. Finally, adding the ability to detect mixed breeds like puggles or labradoodles would be useful since these designer breeds are quite common. 