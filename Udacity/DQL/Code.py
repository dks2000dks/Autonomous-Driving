from DeepQLearning import DQL
from DeepQNetwork import DQN
import torch as tr
import segmentation_models_pytorch as smp
import numpy as np

Learn = DQL(
	DQN=DQN,
	LaneDetectionModel=tr.load("Models/LaneDetection_Model.pth"),
	Preprocessing_Function=smp.encoders.get_preprocessing_fn("efficientnet-b0", "imagenet"),
	Weights_Path="Models",
	Batch_Size=16,
	random_frames=25000,
	greedy_frames=500000,
	max_memory=100000,
	Steps_per_Model_Update=4,
	Steps_per_TargetModel_Update=100,
	discount_rate=0.99,
	epsilon=1.0,
	min_epsilon=0.1,
	max_epsilon=1.0
)

i = 0
while(1):
	i+=1

	image = np.random.randn(160,320,3)
	speed = np.random.uniform(0,10,1)[0]
	throttle = np.random.uniform(0,1,1)[0]
	steering_angle = np.random.uniform(-1,1,1)[0]
	
	action_throttle, action_steering_angle = Learn.getActions(image, speed, throttle, steering_angle)
	print ("i = " + str(i) + ", Applied Throttle = " + str(action_throttle) + ", Steering Angle = " + str(action_steering_angle))