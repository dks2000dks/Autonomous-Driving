# %% [markdown]
# # Drive Udacity Simulator
# 
# Driving Udacity Simulator using Python

# %% [markdown]
# ## Importing Libraries

# %%
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

# Libraries for IO Operations
import argparse
import base64
from datetime import datetime
import shutil
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from io import BytesIO

from DeepQLearning import DQL
from DeepQNetwork import DQN

# PyTorch Libraries
import torch as tr
from torch.utils.data import DataLoader, Dataset
from segmentation_models_pytorch.utils import base
from torchinfo import summary
import segmentation_models_pytorch as smp

# %% [markdown]
# ## Connecting to Server

# %%
sio = socketio.Server()
app = Flask(__name__)

# %% [markdown]
# ## Controller

# %%
class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)

# %% [markdown]
# ## Connect and Send Control

# %%
def sendControl(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("Connection Established")
    sendControl(0, 0)

# %% [markdown]
# ## Telemetry

# %%
@sio.on('telemetry')
def telemetry(sid, data):
	# Current steering angle of the car
	steering_angle = float(data["steering_angle"])

	# Current throttle of the car
	throttle = float(data["throttle"])

	# Current speed of the car
	speed = float(data["speed"])

	# Current image from the center camera of the car
	imgString = data["image"]
	image = Image.open(BytesIO(base64.b64decode(imgString)))
	image = np.asarray(image)

	# Estimating actions
	action_throttle, action_steering_angle = DQL.getActions(image, speed, throttle, steering_angle)

	# Sending control
	sendControl(action_steering_angle, action_throttle)

# %% [markdown]
# ## Starting

# %% [markdown]
# Importing and Setting Parameters

# %%
# Model Path
Model_Path = "Models/LaneDetection_Model.pth"

# Deep Q Learning
Learn = DQL(
	DQN=DQN,
	LaneDetectionModel=tr.load(Model_Path),
	Preprocessing_Function=smp.encoders.get_preprocessing_fn("efficientnet-b0", "imagenet"),
	Weights_Path="Models",
	Batch_Size=16,
	random_frames=25000,
	greedy_frames=500000,
	max_memory=100000,
	Steps_per_Model_Update=4,
	Steps_per_TargetModel_Update=10000,
	discount_rate=0.99,
	epsilon=1.0,
	min_epsilon=0.1,
	max_epsilon=1.0
)

# %% [markdown]
# Connecting to Servers

# %%
# Wrap Flask application with engineio's middleware
app = socketio.Middleware(sio, app)
print ("Wrap Flask application with engineio's middleware: Completed\n")

# Deploy as an eventlet WSGI server
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
print ("Deploy as an eventlet WSGI server: Completed\n")


