import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class DQN(nn.Module):
	def __init__(self):
		super().__init__()
		# Convolutional Layers
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), stride=(2,2), padding=(2,2))
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(5,5), stride=(2,2), padding=(2,2))
		self.conv3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1))

		# Fully Connected Layers
		self.dropout = nn.Dropout()
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(in_features=3872, out_features=512)
		self.fc2 = nn.Linear(in_features=512, out_features=64)
		self.fc31 = nn.Linear(in_features=64, out_features=5)
		self.fc32 = nn.Linear(in_features=64, out_features=5)

	def forward(self, X):
		x = F.elu(self.conv1(X))
		x = F.elu(self.conv2(x))
		x = F.elu(self.conv3(x))
		x = self.dropout(x)
		x = self.flatten(x)
		
		x = F.elu(self.fc1(x))
		x = F.elu(self.fc2(x))

		# Outputs
		y0 = self.fc31(x)
		y1 = self.fc32(x)
		return y0, y1
