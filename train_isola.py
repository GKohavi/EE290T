## Imports ##
#------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.autograd as autograd
#------------------------------------------------#
import matplotlib.pyplot as plt
import numpy as np
from time import time
import math
import os
import sys
from tqdm import tqdm
#------------------------------------------------#
from data import Dataset
#------------------------------------------------#
## Hyperparamters ##
batch_size  = 256			# Number of images to load at a time
epochs 		= 100			# Number of interations throough training data
train_gpu 	= True
lr 		    = 1e-4
train 		= __name__ == "__main__"
## Setting up ##
torch.manual_seed(1)
use_cuda = torch.cuda.is_available() and train_gpu
device = torch.device("cuda" if use_cuda else "cpu")
print('Device mode: ', device)
# ==================Definition Start======================
class SiameseNetwork(nn.Module):
	def __init__(self):
		super(SiameseNetwork, self).__init__()
		self.cnn1 = nn.Sequential(
			nn.Conv2d(3, 8, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(8),
			
			nn.Conv2d(8, 8, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(8),


			nn.Conv2d(8, 8, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(8),
		)
		self.fc1 = nn.Sequential(
			nn.Linear(2*8*15*15, 500),
			nn.LeakyReLU(inplace=True),

			nn.Linear(500, 250),
			nn.LeakyReLU(inplace=True),

			nn.Linear(250, 250),
			nn.LeakyReLU(inplace=True),

			nn.Linear(250, 1),
			nn.Sigmoid())

	def forward_once(self, x):
		output = self.cnn1(x)
		return output

	def forward(self, input1, input2):
		output1 = self.forward_once(input1)
		output2 = self.forward_once(input2)
		output1 = output1.view(output1.size()[0], -1)
		output2 = output2.view(output2.size()[0], -1)
		output = torch.cat([output1, output2], dim=1)
		output = self.fc1(output)
		return output
class ContrastiveLoss(torch.nn.Module):
	def __init__(self, margin=2.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		euclidean_distance = F.pairwise_distance(output1, output2)
		loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
									  (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
		return loss_contrastive
if train == True:
	## Training Data ##
	trainset = Dataset(train=True, split_size=.8)
	trainloader = DataLoader(dataset=trainset, shuffle=True, batch_size=batch_size)

	model = SiameseNetwork().to(device)
	model.train()

	optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))
	criterion = nn.BCELoss()

	for epoch in range(epochs):
		avg_loss = []
		for i, (a, b, c) in tqdm(enumerate(trainloader)):
			a, b, c = (a.permute(0,3,1,2).to(device).type(torch.float32), 
					   b.permute(0,3,1,2).to(device).type(torch.float32), c.to(device).type(torch.float)) #for contrastive loss it is 0 for matching pairs	
			model.zero_grad()
			pred_c = model(a[:,:3,:,:], b[:,:3,:,:])
			loss = criterion(pred_c, c.unsqueeze(1))
			loss.backward()
			optimizer.step()
			avg_loss.append(loss.item())
		print('epoch', epoch, "loss", np.mean(np.array(avg_loss)))

	torch.save(model, 'model_isola_nodepth')
