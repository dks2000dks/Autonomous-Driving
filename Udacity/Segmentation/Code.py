# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import imageio
from PIL import Image, ImageFilter
import cv2
import scipy
import albumentations as albu
import os
import random
import re

# Importing Utils
from Utils import KMeans

# PyTorch Libraries
#!pip3 install segmentation-models-pytorch torchinfo
import torch as tr
from torch.utils.data import DataLoader, Dataset
from segmentation_models_pytorch.utils import base
from torchinfo import summary
import segmentation_models_pytorch as smp

# Loading Model
Model = tr.load("Model/Segmentation_Model.pth")

# Preprocessing Function
Preprocessing_Function = smp.encoders.get_preprocessing_fn("resnet34", "imagenet")

# Test Images
Path = "Images"
Test_Images = os.listdir(Path)
Test_Images.sort()

# Generating Video
video1 = cv2.VideoWriter("Original.mkv", 0, 5, (320, 160))
video2 = cv2.VideoWriter("Edges.mkv", 0, 5, (320, 160))

for f in Test_Images:
	# Importing and Resizing
	image = imageio.imread(Path + "/" + f)
	video1.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

	image = Preprocessing_Function(image)
	image = np.transpose(image, (2,0,1)).astype("float32")

	# Predicting
	X = tr.from_numpy(np.expand_dims(image, axis=0)).to("cuda")
	pred = Model.predict(X).cpu().numpy()[0]

	# Reshape image and prediction back to their original shapes
	image = image.transpose(1,2,0)
	pred = np.argmax(pred, axis=0)

	# Detecting Edges
	pred = (pred == 7).astype(float)
	pred = cv2.erode(pred, np.ones((5,5), np.uint8), iterations=1)
	pred = np.uint8(np.clip(pred*255.0, 0, 255))

	# Detecting Edges
	pred = Image.fromarray(pred)
	# pred = pred.filter(ImageFilter.FIND_EDGES)
	
	# For Generating Video
	overlap = np.zeros((160,320,3))
	overlap[:,:,2] = pred
	overlap = np.uint8(overlap)

	output = cv2.cvtColor(imageio.imread(Path + "/" + f), cv2.COLOR_RGB2BGR)
	overlap = cv2.cvtColor(np.array(overlap), cv2.COLOR_RGB2BGR)
	cv2.addWeighted(overlap, 10, output, 1, 0, output)
	
	video2.write(np.array(output))

	"""
	# Displaying Image
	plt.figure(figsize=(12,6))

	plt.subplot(1,2,1)
	plt.title("Input Image")
	plt.imshow(image)

	plt.subplot(1,2,2)
	plt.title("Predicted Mask")
	plt.imshow(pred, cmap='gray')

	plt.show()
	"""

video1.release()
video2.release()