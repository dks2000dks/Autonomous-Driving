import torch as tr
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
	def __init__(self):
		super().__init__()
		# Convolutional Layers
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5,5), stride=(2,2), padding=(2,2))
		self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5,5), stride=(2,2), padding=(2,2))
		self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5,5), stride=(2,2), padding=(2,2))
		self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
		self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))

		# Fully Connected Layers
		self.dropout = nn.Dropout()
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(in_features=51200, out_features=100)
		self.fc2 = nn.Linear(in_features=100, out_features=50)
		self.fc31 = nn.Linear(in_features=50, out_features=11)
		self.fc32 = nn.Linear(in_features=50, out_features=21)

	def forward(self, X):
		x = F.elu(self.conv1(X))
		x = F.elu(self.conv2(x))
		x = F.elu(self.conv3(x))
		x = F.elu(self.conv4(x))
		x = F.elu(self.conv5(x))
		x = self.dropout(x)
		x = self.flatten(x)
		x = F.elu(self.fc1(x))
		x = F.elu(self.fc2(x))

		# Outputs
		y0 = self.fc31(x)
		y1 = self.fc32(x)
		return y0, y1
