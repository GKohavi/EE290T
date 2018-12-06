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
from tqdm import trange
#------------------------------------------------#
from data import Dataset
from train_isola import SiameseNetwork
#------------------------------------------------#
import scipy.spatial as sp 
import scipy.cluster.hierarchy as hc
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns

import networkx as nx
from sklearn.cluster import SpectralClustering, KMeans
#------------------------------------------------#
model = torch.load('model_isola').cuda()

#Testing
batch_size = 256
accuracy = []
trainset = Dataset(train=False, split_size=.8, patches=False)
ts = trainset
trainloader = DataLoader(dataset=trainset, batch_size=batch_size)
# Dataset intrinsics
skip_len = trainset.skip_len-1
ps = trainset.ps
shape = trainset.shape

model.eval()
def get_neighbors(x, y, i, j):
    xyc = [(x+skip_len, y, i+1, j), (x-skip_len, y, i-1, j), (x+skip_len, y+skip_len, i+1, j+1), 
           (x-skip_len, y+skip_len, i-1, j+1), (x+skip_len, y-skip_len, i+1, j-1), 
           (x-skip_len, y-skip_len, i-1, j+1), (x, y+skip_len, i, j+1), (x, y-skip_len, i, j-1),
           (x+skip_len*2, y, i+2, j),
           (x-skip_len*2, y-skip_len*2, i-2, j+2), 
           (x+skip_len*2, y+skip_len*2, i+2, j+2)]
    xyc = filter(lambda x: ps<x[0]<shape[0]-ps and 
                              ps<x[1]<shape[1]-ps, xyc)
    xyc = list(xyc)
    return xyc
def np_img2torch(img):
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    img = img.cuda()
    return img
def torch_img2np(img):
    img = img.permute(0, 2, 3, 1)
    img = img.detach().cpu().numpy()
    img = img.squeeze(0)
    return img

dists = np.ones((32*43, 32*43))
# adj_matrix = {}
print('starting')
with torch.no_grad():
    # Construct tree
    img_ = trainset[2]
    for i in [3]: # todo, didn't actually train with all chanels normalized
        img_[:,:,i] = (img_[:,:,i]-np.mean(img_[:,:,i])) / np.std(img_[:,:,i])
    start = time()
    img_ = np_img2torch(img_)
    lst_len = 0
    for Ai, Ax in enumerate(trange(ps, shape[0], skip_len)):
        for Aj, Ay in enumerate(range(ps, shape[1], skip_len)):
            lst_len += 1
            A = ts.get_from_xy(Ax, Ay, img_)
            A_neighbors = get_neighbors(Ax, Ay, Ai, Aj)
            for Bx, By, Bi, Bj in A_neighbors:
                # print(Ax, Ay, Ai, Aj, Bx, By, Bi, Bj)
                B = ts.get_from_xy(Bx, By, img_)
                if A.shape != B.shape: # happens when reading corner
                    C = 0
                else:
                    C = model(A, B)[0][0].item()
                # C = 1 - C
                dists[Ai*Aj+Aj, Bi*Bj+Bj] = C
    end = time()
    print(end-start)
    # dists=np.array(dists)  
    print(lst_len)
    print(Ai, Aj)
    print(dists.shape)

    # plt.figure()
    # plt.hist(dists)
    # plt.show()

    sc = SpectralClustering(12, affinity='precomputed', n_init=100)
    sc.fit(dists)
    print('spectral clustering')
    print(sc.labels_)

    # plt.figure()
    # plt.hist(sc.labels_)
    # plt.show()

    customPalette = [np.array([220,20,60]), 
                     np.array([255,69,0]), 
                     np.array([255,215,0]), 
                     np.array([240,230,140]), 
                     np.array([154,205,50]),
                     np.array([85,107,47]),
                     np.array([124,252,0]),
                     np.array([0,128,128]),
                     np.array([224,255,255]),
                     np.array([70,130,180]),
                     np.array([75,0,130]),
                     np.array([139,0,139]),
                     ]
    customPalette = [[cc/255 for cc in c] for c in customPalette]
    idx = 0
    img_ = torch_img2np(img_)
    img_2 = np.copy(img_)
    for Ai, Ax in enumerate(range(ps, shape[0], skip_len)):
        for Aj, Ay in enumerate(range(ps, shape[1], skip_len)):
            img_[Ax-ps:Ax+ps+1,Ay-ps:Ay+ps+1,:3] = customPalette[sc.labels_[idx]%12] 
            idx+=1
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img_[:,:,:3])
    plt.subplot(1, 2, 2)
    plt.imshow(img_2[:,:,:3])
    plt.show()

