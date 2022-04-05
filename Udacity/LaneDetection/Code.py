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
Model = tr.load("Model/LaneDetection_Model.pth")

# Preprocessing Function
Preprocessing_Function = smp.encoders.get_preprocessing_fn("efficientnet-b0", "imagenet")

# Test Images
Path = "Images"
Test_Images = os.listdir(Path)
Test_Images.sort()

# Generating Video
video1 = cv2.VideoWriter("Original.mkv", 0, 5, (320, 160))
video2 = cv2.VideoWriter("Edges.mkv", 0, 5, (320, 160))
cm = plt.get_cmap('jet')

for f in Test_Images:
	# Importing and Resizing
	image = imageio.imread(Path + "/" + f)
	#image = cv2.resize(image, (320, 160))
	I = image
	video1.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

	image = Preprocessing_Function(image)
	image = np.transpose(image, (2,0,1)).astype("float32")

	# Predicting
	X = tr.from_numpy(np.expand_dims(image, axis=0)).to("cuda")
	pred = Model.predict(X).cpu().numpy()[0]

	# Reshape image and prediction back to their original shapes
	image = image.transpose(1,2,0)
	pred = np.argmax(pred, axis=0)
	pred = np.eye(3)[pred]
	pred[:,:,0] = np.zeros(pred[:,:,0].shape)
	pred = (pred*255.0).astype(np.uint8)

	# Generaring Video
	output = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
	overlap = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
	cv2.addWeighted(overlap, 10, output, 1, 0, output)

	video2.write(output)

video1.release()
video2.release()
