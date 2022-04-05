# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import imageio
import cv2
import scipy
import albumentations as albu
import os
import random
import re

# PyTorch Libraries
#!pip3 install segmentation-models-pytorch torchinfo
import torch as tr
from torch.utils.data import DataLoader, Dataset
from segmentation_models_pytorch.utils import base
from torchinfo import summary
import segmentation_models_pytorch as smp

# Loading Model
Model = tr.load("LaneDetection_Model.pth")

# Preprocessing Function
Preprocessing_Function = smp.encoders.get_preprocessing_fn("efficientnet-b0", "imagenet")

# Test Images
Path = "Test_Images"
Test_Images = os.listdir(Path)
for f in Test_Images:
	# Importing and Resizing
	image = imageio.imread(Path + "/" + f)
	image = cv2.resize(image, (512,256))
	image = Preprocessing_Function(image)
	#image = image/255.0
	image = np.transpose(image, (2,0,1)).astype("float32")

	# Predicting
	X = tr.from_numpy(np.expand_dims(image, axis=0)).to("cuda")
	pred = Model.predict(X).cpu().numpy()[0]

	# Displaying Image
	plt.figure(figsize=(12,6))

	plt.subplot(1,2,1)
	plt.title("Input Image")
	plt.imshow(image.transpose(1,2,0))

	plt.subplot(1,2,2)
	plt.title("Predicted Mask")
	plt.imshow(np.argmax(pred, axis=0))

	plt.show()
