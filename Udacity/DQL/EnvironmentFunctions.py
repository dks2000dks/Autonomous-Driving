import numpy as np
import torch as tr

# Designed Reward Function
def Reward_Function(State, Speed, LaneDetectionModel, Preprocessing_Function):
	# Reward for Speed
	maxSpeed = 15
	if Speed <= maxSpeed*0.5:
		Speed_Reward = Speed/maxSpeed*0.5
	elif Speed > maxSpeed*0.5 and Speed <= 0.75*maxSpeed:
		Speed_Reward = 3 - 4*(Speed/maxSpeed)
	else:
		Speed_Reward = 1 - (Speed/0.75*maxSpeed)

	# Reward for Lane

	# Detecting Lane
	image = Preprocessing_Function(State)
	image = np.transpose(image, (2,0,1)).astype("float32")
	X = tr.from_numpy(np.expand_dims(image, axis=0)).cuda()
	pred = LaneDetectionModel(X).cpu().detach().numpy()[0]
	pred = np.argmax(pred, axis=0)
	
	if np.unique(pred).shape[0] == 3:
		r,c = np.where(pred == 1)
		lr,lc = r[-1], c[-1]
		r,c = np.where(pred == 2)
		rr,rc = r[-1], c[-1]
		Lane_Reward = (140 - lc) + (rc - 180)
	elif np.unique(pred).shape[0] == 2:
		if np.unique(pred)[-1] == 1:
			r,c = np.where(pred == 1)
			lr,lc = r[-1], c[-1]
			Lane_Reward = (140 - lc)
		else:
			r,c = np.where(pred == 2)
			rr,rc = r[-1], c[-1]
			Lane_Reward = (rc - 180)
	else:
		Lane_Reward = 0
	
	return Speed_Reward + Lane_Reward

# Designed Crash Scenario
def Crash_Scenario(Speed, Throttle):
	Speed = np.array(Speed)[:-10]
	Throttle = np.array(Throttle)[:-10]

	if np.all(Throttle > 0.8) and np.all(Speed < 1):
		return 1
	else:
		return 0