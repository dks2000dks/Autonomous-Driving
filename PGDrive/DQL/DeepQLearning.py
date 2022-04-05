import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm,trange

# Importing Deep-Q-Network
from DeepQNetwork import DQN

# Environment
import gym
import pgdrive
from pgdrive import PGDriveEnv

# PyTorch Libraries
import torch as tr
import torch.nn as nn
import torch.nn.functional as F

# Deep Q Learning
class DQL():
	def __init__(self,
		env,
		DQN,
		Weights_Path,
		Batch_Size,
		Episodes,
		TimeSteps,
		greedy_frames,
		max_memory,
		Steps_per_Model_Update,
		Steps_per_TargetModel_Update,
		Parameters_Path,
		skip_frames=3,
		discount_rate=0.99,
		epsilon=1.0,
		min_epsilon=0.05,
		max_epsilon=1.0):

		# Assigning Variable
		self.env = env
		self.Weights_Path = Weights_Path
		self.Episodes = Episodes
		self.TimeSteps = TimeSteps
		self.Batch_Size = Batch_Size
		self.greedy_frames = greedy_frames
		self.max_memory = max_memory
		self.Steps_per_Model_Update = Steps_per_Model_Update
		self.Steps_per_TargetModel_Update = Steps_per_TargetModel_Update
		self.skip_frames = skip_frames
		self.discount_rate = discount_rate
		self.min_epsilon = min_epsilon
		self.max_epsilon = max_epsilon

		# Checking for Trained Model
		if os.path.isfile(self.Weights_Path + "/Model.pth") == False:
			self.Model = DQN()
			self.Target_Model = DQN()
			tr.save(self.Model, self.Weights_Path + "/Model.pth")
			tr.save(self.Target_Model, self.Weights_Path + "/Target_Model.pth")
		self.Model = tr.load(self.Weights_Path + "/Model.pth")
		self.Target_Model = tr.load(self.Weights_Path + "/Target_Model.pth")

		# Loading for Parameters
		self.Parameters_Path = Parameters_Path

		# Device
		self.device = tr.device("cuda")
		
		# Move model to Device
		self.Model.to(self.device)
		self.Target_Model.to(self.device)

		# Model Training Settings
		self.criterion1 = tr.nn.HuberLoss()
		self.criterion2 = tr.nn.HuberLoss()
		self.optimizer = tr.optim.Adam(self.Model.parameters())
		
		# Actions
		self.Actions = np.array([-0.5, -0.25, 0, 0.25, 0.5])
		self.Num_Actions = self.Actions.shape[0]
	
	
	def LoadParameters(self):
		# Epsilon
		if os.path.isfile(self.Parameters_Path + "/Epsilon.npy") == False:
			self.epsilon = 1.0
		else:
			self.epsilon = np.load(self.Parameters_Path + "/Epsilon.npy")[0]
		
		# frame_count  
		if os.path.isfile(self.Parameters_Path + "/frame_count.npy") == False:
			self.frame_count = 0
		else:
			self.frame_count = np.load(self.Parameters_Path + "/frame_count.npy")[0]

		# memory_index
		if os.path.isfile(self.Parameters_Path + "/memory_index.npy") == False:
			self.memory_index = 0
		else:
			self.memory_index = np.load(self.Parameters_Path + "/memory_index.npy")[0]

		# Episode Rewards
		if os.path.isfile(self.Parameters_Path + "/Episodes_Rewards.npy") == False:
			self.Episodes_Rewards = []
		else:
			self.Episodes_Rewards = list(np.load(self.Parameters_Path + "/Episodes_Rewards.npy"))
			
		# Episode Count
		if os.path.isfile(self.Parameters_Path + "/Episodes_Count.npy") == False:
			episode = 0
		else:
			episode = list(np.load(self.Parameters_Path + "/Episodes_Count.npy"))[0]
			
		# Printing Parameters
		print ("\n" + "-"*65)
		print ("Epsilon = " + str(self.epsilon) + ", Frame Count = " + str(self.frame_count) + ", Memory Index = " + str(self.memory_index) + ", Episodes = " + str(episode))
		print ("")
		
		return episode


	def SaveParameters(self, episode):
		# Epsilon
		np.save(self.Parameters_Path + "/Epsilon.npy", np.array([self.epsilon]))
		
		# frame_count  
		np.save(self.Parameters_Path + "/frame_count.npy", np.array([self.frame_count]))

		# memory_index
		np.save(self.Parameters_Path + "/memory_index.npy", np.array([self.memory_index]))

		# Episode Rewards
		np.save(self.Parameters_Path + "/Episodes_Rewards.npy", np.array(self.Episodes_Rewards))
		
		# Episode Count
		np.save(self.Parameters_Path + "/Episodes_Count.npy", np.array([episode+1]))
		
		
	def getBatch(self, Indices):
		# Getting a Batch
		State_Samples = []
		Next_State_Samples = []
		Action_Steering_Angle_Samples = []
		Action_Throttle_Samples = []
		Rewards_Samples = []
		Done_Samples = []
		
		# Loading Data
		for i in Indices:
			# States
			State_Samples.append(np.load(self.Parameters_Path + "/state/" + str(i) + ".npy"))
			Next_State_Samples.append(np.load(self.Parameters_Path + "/next_state/" + str(i) + ".npy"))
			
			# Actions
			Action_Steering_Angle_Samples.append(np.load(self.Parameters_Path + "/action_steering_angle/" + str(i) + ".npy")[0])
			Action_Throttle_Samples.append(np.load(self.Parameters_Path + "/action_throttle/" + str(i) + ".npy")[0])
			
			# Rewards and Done
			Rewards_Samples.append(np.load(self.Parameters_Path + "/reward/" + str(i) + ".npy")[0])
			Done_Samples.append(np.load(self.Parameters_Path + "/done/" + str(i) + ".npy")[0])
			
		return tr.Tensor(np.array(State_Samples)), tr.Tensor(np.array(Next_State_Samples)), tr.Tensor(np.array(Action_Steering_Angle_Samples)), tr.Tensor(np.array(Action_Throttle_Samples)), tr.Tensor(np.array(Rewards_Samples)), tr.Tensor(np.array(Done_Samples))


	def saveSample(self, state, next_state, action_steering_angle, action_throttle, reward, done):
		# Saving Samples
		
		# States
		np.save(self.Parameters_Path + "/state/" + str(self.memory_index) + ".npy", state)
		np.save(self.Parameters_Path + "/next_state/" + str(self.memory_index) + ".npy", next_state)
		
		# Actions
		np.save(self.Parameters_Path + "/action_steering_angle/" + str(self.memory_index) + ".npy", np.array([action_steering_angle]))
		np.save(self.Parameters_Path + "/action_throttle/" + str(self.memory_index) + ".npy", np.array([action_throttle]))
		
		# Reward and Done
		np.save(self.Parameters_Path + "/reward/" + str(self.memory_index) + ".npy", np.array([reward]))
		np.save(self.Parameters_Path + "/done/" + str(self.memory_index) + ".npy", np.array([done]))
		

	
	def Train(self):
		episode = self.LoadParameters()
		print ("-"*25 + "Starting Training" + "-"*25)
		print ("\n")

		for episode in range(episode, self.Episodes):
			state = self.env.reset()['image'].transpose(1,0,2)	

			episode_reward = 0
			pbar = trange(self.TimeSteps, desc="Episode-" + str(episode), unit="")

			for t in pbar:
				# env.render()
				
				# Epsilon-Decay with increase of no.of Frames => Decaying Probability of Random Action
				self.epsilon = self.min_epsilon + ((self.max_epsilon - self.min_epsilon) * np.exp(-self.frame_count/self.greedy_frames))
				self.epsilon = max(self.epsilon, self.min_epsilon)
				self.frame_count += 1

				"""
				Epsilon-Greedy Algorithm
				"""
				if self.epsilon > np.random.rand(1)[0]:
					# Epsilon Step
					action_steering_angle = self.Actions[np.random.randint(self.Num_Actions)]
					action_throttle = self.Actions[np.random.randint(self.Num_Actions)]
				else:
					# Greedy Step => Perform Action with most Q-Value
					with tr.no_grad():
						I = tr.unsqueeze(tr.permute(tr.Tensor(state), (2,0,1)), axis=0)
						y0, y1 = self.Model(I.cuda())
						y0, y1 = y0[0].cpu().detach(), y1[0].cpu().detach()
					action_steering_angle = -0.5 + (0.25 * tr.argmax(y0))
					action_throttle = -0.5 + (0.25 * tr.argmax(y1))
				
				# Rounding off
				action_steering_angle = np.round(action_steering_angle, decimals=2)
				action_throttle = np.round(action_throttle, decimals=2)

				# Applying Action to the Environment
				next_state, reward, done, info = self.env.step([action_steering_angle, action_throttle])
				next_state = np.array(next_state['image']).transpose(1,0,2)

				# Increasing Episode Reward
				episode_reward += reward
				pbar.set_postfix(reward = reward)

				# Updating Data in Replay
				if self.frame_count % self.skip_frames == 0:
					# Adding Replay Memory
					self.saveSample(state, next_state, action_steering_angle, action_throttle, reward, done)
						
					# Updating Memory Index
					self.memory_index = (self.memory_index + 1) % self.max_memory
				
				# Setting Next State as Current State
				state = next_state
				
				# Size of Memory
				Memory_Size = len(os.listdir(self.Parameters_Path + "/state"))
							

				# Updating the Model
				if self.frame_count % self.Steps_per_Model_Update == 0 and Memory_Size > self.Batch_Size:

					# Samples to Train
					Indices = np.random.choice(Memory_Size, size=self.Batch_Size, replace=False)

					State_Samples, Next_State_Samples, Action_Steering_Angle_Samples, Action_Throttle_Samples, Rewards_Samples, Done_Samples = self.getBatch(Indices)					

					"""
					s		: State
					a		: Action
					s'		: Next_State
					r		: Reward
					gamma	: discount_rate
					"""

					# max_(a') Q(s',a',TargetModel)
					with tr.no_grad():
						pred_y0, pred_y1 = self.Target_Model(tr.permute(Next_State_Samples, (0,3,1,2)).cuda())
						pred_y0, pred_y1 = pred_y0.cpu(), pred_y1.cpu()
					
					future_sterring_angle_rewards = tr.max(pred_y0, axis=1).values
					future_throttle_rewards = tr.max(pred_y1, axis=1).values

					# y = r + gamma * max_(a') Q(s',a',TargetModel)
					y0 = Rewards_Samples + self.discount_rate * future_sterring_angle_rewards
					y1 = Rewards_Samples + self.discount_rate * future_throttle_rewards

					# Q value of Last-Frame is set to "-1" => When it is last frame before done
					y0 = y0 * (1 - Done_Samples) - Done_Samples
					y1 = y1 * (1 - Done_Samples) - Done_Samples

					# Masks of Actions
					Masks_Steering_Angle = F.one_hot((2+4*Action_Steering_Angle_Samples).type(tr.int64), self.Num_Actions)
					Masks_Throttle = F.one_hot((2+4*Action_Throttle_Samples).type(tr.int64), self.Num_Actions)

					## Training
					self.optimizer.zero_grad()
					
					# Q(s,Model)
					Q_Values0, Q_Values1 = self.Model(tr.permute(State_Samples, (0,3,1,2)).cuda())
					Q_Values0, Q_Values1 = Q_Values0.cpu(), Q_Values1.cpu()
					
					# Q(s,a,Model)
					y_pred0 = tr.sum(tr.multiply(Q_Values0, Masks_Steering_Angle), axis=1)
					y_pred1 = tr.sum(tr.multiply(Q_Values1, Masks_Throttle), axis=1)

					# Loss
					loss = self.criterion1(y_pred0, y0) + self.criterion2(y_pred1, y1)
					
					# Updating Model
					loss.backward()
					self.optimizer.step()

				# Updating Target Model
				if self.frame_count % self.Steps_per_TargetModel_Update == 0:
					tr.save(self.Model, self.Weights_Path + "/Model.pth")
					self.Target_Model = tr.load(self.Weights_Path + "/Model.pth")

				# Done
				if done:
					break

			# Updating Episodes Rewards
			tr.save(self.Model, self.Weights_Path + "/Model.pth")
			tr.save(self.Target_Model, self.Weights_Path + "/Target_Model.pth")
			print ("Episode-" + str(episode) + ": Episode-Reward = " + str(episode_reward) + ", No.of frames: " + str(self.frame_count) + ", Memory size: " + str(Memory_Size) + "\n")
			self.Episodes_Rewards.append(episode_reward)

			# Save Parameters
			self.SaveParameters(episode)

			if len(self.Episodes_Rewards) > 50:
				del self.Episodes_Rewards[:1]
			
			if np.mean(self.Episodes_Rewards) > 250:
				tr.save(self.Model, self.Weights_Path + "/Model.pth")
				print ("Solved at Episode-{}".format(episode))
				break


# Configuration of Environment
config = dict(
	# Rendering
	use_render=False,
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
env.reset()

# Deep Q Learning
Learn = DQL(
	env=env,
	DQN=DQN,
	Weights_Path="Models",
	Batch_Size=32,
	Episodes=5000,
	TimeSteps=int(5*1e4),
	greedy_frames=int(5*1e6),
	max_memory=int(3*1e4),
	Steps_per_Model_Update=int(4),
	Steps_per_TargetModel_Update=int(3*1e3),
	Parameters_Path = "Data"
)

# Training Model
Learn.Train()
