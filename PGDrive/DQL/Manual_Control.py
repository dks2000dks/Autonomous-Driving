import numpy as np
import matplotlib.pyplot as plt
import gym
import pgdrive
from pgdrive import PGDriveEnv
import warnings
warnings.filterwarnings("ignore")

# Configuration of Environment
config = dict(
	# Rendering
	use_render=True,
	manual_control=True,
	use_chase_camera=True,

	# Traffic
	traffic_density=0.0,

	# Obseravation
	use_image=True,
	rgb_clip=True,
	vehicle_config=dict(show_navi_mark=False, rgb_cam=(256,192)),
	image_source='rgb_cam',

	# Map
	map='CCC',
	controller="keyboard"
	)

# Creating Environment
env = PGDriveEnv(config)
state = env.reset()

# Visualisations
for i in range(int(1e5)):
	obs, reward, done, info = env.step(env.action_space.sample())
	env.render()
	if done:
		env.reset()

# Environment Spaces
print ("Action Spaces of Environment:", env.action_space)
print ("Observations:", obs.keys())
print ("Observed Image Shape:", obs['image'].shape)
