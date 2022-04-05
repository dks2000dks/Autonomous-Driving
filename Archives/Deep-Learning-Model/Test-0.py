# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import scipy
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Tensorflow Libraries
import tensorflow as tf
from tensorflow.keras.models import load_model

# Loading Pre-Trained Lane-Detection Model
Model = load_model("model.h5")
print (Model.summary())

# Reading Image
img = pimg.imread('Images/Image.jpg')
Original_Shape = img.shape

# Displaying Image
plt.figure(figsize=(6,6))
plt.title("Original Image")
plt.imshow(img)
plt.show()

# Resizing Image
I = cv2.resize(img, (160,80))

# Displaying Image
plt.figure(figsize=(6,6))
plt.title("Resized Image")
plt.imshow(I)
plt.show()

# Predicting Lane
Lane = Model.predict(np.expand_dims(I,axis=0))[0]

# Resizing Image
Lane = cv2.resize(Lane, (Original_Shape[1], Original_Shape[0]))

# Displaying Image
plt.figure(figsize=(6,6))
plt.title("Lane Image")
plt.imshow(Lane, cmap='gray')
plt.show()
