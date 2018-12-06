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
from train_isola import SiameseNetwork
#------------------------------------------------#
model = torch.load('model_isola_nodepth').cuda()

#Testing
batch_size = 256
accuracy = []
trainset = Dataset(train=False, split_size=.8)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size)

model.eval()
with torch.no_grad():
	for epoch in range(5):
		for i, (a, b, c) in enumerate(trainloader):
			a, b, c = (a.permute(0,3,1,2).cuda().type(torch.float32), 
					   b.permute(0,3,1,2).cuda().type(torch.float32), c.cuda().type(torch.float))
			pred_c = model(a[:,:3,:,:], b[:,:3,:,:]).squeeze(1)
			pred_c = list(torch.round(pred_c).cpu().numpy())
			c = list(c.cpu().numpy())
			accuracy.append(sum([1 for x, y in zip(pred_c, c) if x == y]) / len(c))
	print('Accuracy:', sum(accuracy) / len(accuracy))

