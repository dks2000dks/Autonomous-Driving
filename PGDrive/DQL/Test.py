import numpy as np
import matplotlib.pyplot as plt
import gym
import pgdrive
from pgdrive import PGDriveEnv
import warnings
warnings.filterwarnings("ignore")

# PyTorch Libraries
import torch as tr
import torch.nn as nn
import torch.nn.functional as F

# Configuration of Environment
config = dict(
	# Rendering
	use_render=True,
	manual_control=False,
	use_chase_camera=True,

	# Traffic
	traffic_density=0.0,

	# Obseravation
	use_image=True,
	rgb_clip=True,
	vehicle_config=dict(show_navi_mark=False, rgb_cam=(84,84)),
	image_source='rgb_cam',

	# Map
	map="CS"
	)

# Creating Environment
env = PGDriveEnv(config)
state = np.array(env.reset()['image']).transpose(1,0,2)

# Load Model
device = tr.device("cuda")
Model = tr.load("Models/Model.pth")
Model.to(device)

while(1):
	#v Estimating Actions
	I = tr.unsqueeze(tr.permute(tr.Tensor(state), (2,0,1)), axis=0)
	y0, y1 = Model(I.cuda())
	y0, y1 = y0[0].cpu().detach(), y1[0].cpu().detach()
	action_steering_angle = -0.5 + (0.25 * tr.argmax(y0))
	action_throttle = -0.5 + (0.25 * tr.argmax(y1))

	# Playing
	next_state, reward, done, info = env.step([action_steering_angle, action_throttle])
	next_state = np.array(next_state['image']).transpose(1,0,2)

	# Rendering
	env.render()
	state = next_state

	if done:
		print ("Program Exiting")
		env.reset()
