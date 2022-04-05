import numpy as np
import os

# Importing Deep-Q-Network
from DeepQNetwork import DQN

# Importing Environment Functions
from EnvironmentFunctions import Reward_Function, Crash_Scenario

# PyTorch Libraries
import torch as tr
import torch.nn as nn
import torch.nn.functional as F


# Deep Q Learning
class DQL():
	def __init__(self,
		DQN,
		LaneDetectionModel,
		Preprocessing_Function,
		Weights_Path,
		Batch_Size,
		random_frames,
		greedy_frames,
		max_memory,
		Steps_per_Model_Update,
		Steps_per_TargetModel_Update,
		discount_rate=0.99,
		epsilon=1.0,
		min_epsilon=0.1,
		max_epsilon=1.0):
	
		self.LaneDetectionModel = LaneDetectionModel
		self.Preprocessing_Function = Preprocessing_Function
		self.Weights_Path = Weights_Path
		self.Batch_Size = Batch_Size
		self.random_frames = random_frames
		self.greedy_frames = greedy_frames
		self.max_memory = max_memory
		self.Steps_per_Model_Update = Steps_per_Model_Update
		self.Steps_per_TargetModel_Update = Steps_per_TargetModel_Update
		self.discount_rate = discount_rate
		self.epsilon = epsilon
		self.min_epsilon = min_epsilon
		self.max_epsilon = max_epsilon

		if os.path.isfile(self.Weights_Path + "/Model.pth"):
			self.Model = tr.load(self.Weights_Path + "/Model.pth")
			self.Target_Model = tr.load(self.Weights_Path + "/Model.pth")
		else:
			self.Model = DQN()
			self.Target_Model = DQN()
			tr.save(self.Model, self.Weights_Path + "/Model.pth")


		# Replay Memory
		self.Replay = {}

		History = ["state", "speed", "throttle", "action_throttle", "action_steering_angle", "next_state", "next_speed", "next_throttle", "reward", "done"]
		for p in History:
			self.Replay[p] = []


		# Previous Data
		self.previous_state = None
		self.previous_speed = None
		self.previous_throttle = None
		self.previous_action_throttle = None
		self.previous_action_steering_angle = None

		# Count
		self.frame_count = -1
		self.episodes = 0
		self.episode_reward = 0
		self.episode_rewards = []

		# Device
		self.device = tr.device("cuda")
		
		# Move model to Device
		self.Model.to(self.device)
		self.Target_Model.to(self.device)


	def getActions(self, image, speed, throttle, steering_angle):
		
		self.frame_count += 1

		if self.frame_count < self.random_frames:
			action_throttle = 0.5*np.random.choice([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
			action_steering_angle = 0.5*np.random.choice([-1. , -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0. , 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ])
		else:
			"""
			Epsilon-Greedy Algorithm
			"""
			if self.epsilon > np.random.rand(1)[0]:
				# Epsilon Step
				action_throttle = 0.5*np.random.choice([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
				action_steering_angle = 0.5*np.random.choice([-1. , -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0. , 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ])
			else:
				# Greedy Step => Perform Action with most Q-Value
				I = tr.unsqueeze(tr.permute(image, (2,0,1)), axis=0)
				y0, y1 = self.Model(I.cuda())[0]
				y0, y1 = y0.cpu().detach(), y1.cpu().detach()
				action_throttle = 0.5*(0 + (0.1 * tr.argmax(y0)))
				action_steering_angle = 0.5*(-1 + (0.1 * tr.argmax(y1)))

		
		# Epsilon-Decay with increase of no.of Frames => Decaying Probability of Random Action
		self.epsilon -= (self.max_epsilon - self.min_epsilon)/self.greedy_frames
		self.epsilon = max(self.epsilon, self.min_epsilon)

		if self.frame_count == 0:
			self.previous_state = image
			self.previous_speed = speed
			self.previous_throttle = throttle
			self.previous_action_throttle = action_throttle
			self.previous_action_steering_angle = action_steering_angle

			return action_throttle, action_steering_angle


		# Updating Data in Replay
		self.Replay['state'].append(self.previous_state)
		self.Replay['speed'].append(self.previous_speed)
		self.Replay['throttle'].append(self.previous_throttle)

		self.Replay['action_throttle'].append(self.previous_action_throttle)
		self.Replay['action_steering_angle'].append(self.previous_action_steering_angle)

		self.Replay['next_state'].append(image)
		self.Replay['next_speed'].append(speed)
		self.Replay['next_throttle'].append(throttle)


		# Reward and Done
		done = Crash_Scenario(self.Replay["speed"], self.Replay["throttle"])
		if done:
			reward = -100
		else:
			reward = Reward_Function(image, speed, self.LaneDetectionModel, self.Preprocessing_Function)


		# Updating Episode Reward and Updating Replay
		self.episode_reward += reward
		self.Replay['reward'].append(reward)
		self.Replay['done'].append(done)


		# Size of Memory
		Memory_Size = len(self.Replay["state"])


		# Updating the Model
		if self.frame_count % self.Steps_per_Model_Update == 0 and Memory_Size > self.Batch_Size:

			# Samples to Train
			Indices = np.random.choice(Memory_Size, size=self.Batch_Size)

			State_Samples = tr.Tensor(np.array(self.Replay["state"])[Indices])
			Action_Throttle_Samples = tr.Tensor(np.array(self.Replay["action_throttle"])[Indices])
			Action_Steering_Angle_Samples = tr.Tensor(np.array(self.Replay["action_steering_angle"])[Indices])

			Next_State_Samples = tr.Tensor(np.array(self.Replay["next_state"])[Indices])
			Rewards_Samples = tr.Tensor(np.array(self.Replay["reward"])[Indices])
			Done_Samples = tr.Tensor(np.array(self.Replay["done"], dtype=np.float)[Indices])

			"""
			s		: State
			a		: Action
			s'		: Next_State
			r		: Reward
			gamma	: discount_rate
			"""

			# max_(a') Q(s',a',TargetModel)
			pred_y0, pred_y1 = self.Target_Model(tr.permute(Next_State_Samples, (0,3,1,2)).cuda())
			pred_y0, pred_y1 = pred_y0.cpu(), pred_y1.cpu()
			future_throttle_rewards = tr.max(pred_y0, axis=1).values
			future_sterring_angle_rewards = tr.max(pred_y1, axis=1).values

			# y = r + gamma * max_(a') Q(s',a',TargetModel)
			y0 = Rewards_Samples + self.discount_rate * future_throttle_rewards
			y1 = Rewards_Samples + self.discount_rate * future_sterring_angle_rewards

			# Q value of Last-Frame is set to "-1" => When it is last frame before done
			y0 = y0 * (1 - Done_Samples) - Done_Samples
			y1 = y1 * (1 - Done_Samples) - Done_Samples

			# Masks of Actions
			Masks_Throttle = F.one_hot((10*2*Action_Throttle_Samples).type(tr.int64), 11)
			Masks_Steering_Angle = F.one_hot((10+10*2*Action_Steering_Angle_Samples).type(tr.int64), 21)

			# Training
			self.criterion1 = tr.nn.HuberLoss()
			self.criterion2 = tr.nn.HuberLoss()
			self.optimizer = tr.optim.Adam(self.Model.parameters())
			self.optimizer.zero_grad()
			
			# Q(s,Model)
			Q_Values0, Q_Values1 = self.Model(tr.permute(State_Samples, (0,3,1,2)).cuda())
			Q_Values0, Q_Values1 = Q_Values0.cpu(), Q_Values1.cpu()
			
			# Q(s,a,Model)
			y_pred0 = tr.sum(tr.multiply(Q_Values0, Masks_Throttle), axis=1)
			y_pred1 = tr.sum(tr.multiply(Q_Values1, Masks_Steering_Angle), axis=1)

			loss = self.criterion1(y_pred0, y0) + self.criterion2(y_pred1, y1)
			loss.backward()

			self.optimizer.step()

		
		# Updating Target Model
		if self.frame_count % self.Steps_per_TargetModel_Update == 0:
			tr.save(self.Model, self.Weights_Path + "/Model.pth")
			self.Target_Model = tr.load(self.Weights_Path + "/Model.pth")
			

		# Checking Memory
		if Memory_Size > self.max_memory:
			del self.Replay["state"][:1]
			del self.Replay["speed"][:1]
			del self.Replay["throttle"][:1]
			del self.Replay['action_throttle'][:1]
			del self.Replay['action_steering_angle'][:1]
			del self.Replay['next_state'][:1]
			del self.Replay['next_speed'][:1]
			del self.Replay['next_throttle'][:1]
			del self.Replay['reward'][:1]
			del self.Replay['done'][:1]


		# Updating Previous Data
		self.previous_state = image
		self.previous_speed = speed
		self.previous_throttle = throttle
		self.previous_action_throttle = action_throttle
		self.previous_action_steering_angle = action_steering_angle

		return action_throttle, action_steering_angle