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
import scipy.spatial as sp 
import scipy.cluster.hierarchy as hc
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns
#------------------------------------------------#
is_depth = False
if is_depth:
    model = torch.load('model_isola').cuda()
else:
    model = torch.load('model_isola_nodepth').cuda()
model.eval()
#Testing
batch_size = 256
accuracy = []
trainset = Dataset(train=True, split_size=.8, patches=False)
ts = trainset
trainloader = DataLoader(dataset=trainset, batch_size=batch_size)
# Dataset intrinsics
skip_len = trainset.skip_len-1
ps = trainset.ps
shape = trainset.shape


def get_neighbors(x, y):
    xyc = [(x+skip_len, y), (x-skip_len, y), (x+skip_len, y+skip_len), (x-skip_len, y+skip_len),
           (x+skip_len, y-skip_len), (x-skip_len, y-skip_len), (x, y+skip_len), (x, y-skip_len)]
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
dists = []
dists_idx = []
with torch.no_grad():
    # Construct tree
    img_ = trainset[11] #11
    for i in [3]: # todo, didn't actually train with all chanels normalized
        img_[:,:,i] = (img_[:,:,i]-np.mean(img_[:,:,i])) / np.std(img_[:,:,i])
    start = time()
    # img_ = np_img2torch(img_)
    print(img_.shape)
    A_x, A_y = 270, 135#trainset.choose_rand_patch() #270, 135
    print(A_x, A_y)
    show_img_ = np.copy(img_)
    plt.figure()
    plt.imshow(show_img_[A_x-ps:A_x+ps+1, A_y-ps:A_y+ps+1,:][:,:,:3])
    show_img_[A_x-ps:A_x+ps+1, A_y-ps:A_y+ps+1,:] = 0
    plt.figure()
    plt.imshow(show_img_[:,:,:3])
    plt.figure()
    plt.imshow(show_img_[:,:,3])
    plt.show()
    # 1/0
    # plt.show()
    # show how close nieghbors are
    show_img_ = np.copy(img_)
    img_ = np_img2torch(img_)

    A = ts.get_from_xy(A_x, A_y, img_)
    for di, i in enumerate(range(ps, shape[0], skip_len)):
        for dj, j in enumerate(range(ps, shape[1], skip_len)):
            B = ts.get_from_xy(i, j, img_)
            if A.shape != B.shape: # happens when reading corner
                C = 0
            else:
                if is_depth:
                    C = model(A, B)[0][0].item()
                else: 
                    C = model(A[:,:3,:,:], B[:,:3,:,:])[0][0].item()
            # show white
            show_img_[i-ps:i+ps+1, j-ps:j+ps+1,1] *= (1-C) 
            # show_img_[i-ps:i+ps+1, j-ps:j+ps+1,:] = 1 - show_img_[i-ps:i+ps+1, j-ps:j+ps+1,:]
    show_img_[A_x-ps:A_x+ps+1, A_y-ps:A_y+ps+1,0] = 0
    show_img_[A_x-ps:A_x+ps+1, A_y-ps:A_y+ps+1,1] = 1
    show_img_[A_x-ps:A_x+ps+1, A_y-ps:A_y+ps+1,2] = 0
    plt.figure()
    plt.imshow(show_img_[:,:,:3])
    plt.show()



    # for di1, i1 in enumerate(range(ps, shape[0], skip_len)):
    #     for dj1, j1 in enumerate(range(ps, shape[1], skip_len)):
    #         img1 = img_[:,:,i1-ps:i1+ps+1,j1-ps:j1+ps+1]
    #         img1_neighbors = get_neighbors(i1, j1)

    #         for di2, i2 in enumerate(range(ps, shape[0], skip_len)):
    #             for dj2, j2 in enumerate(range(ps, shape[1], skip_len)):
    #                 if (i2, j2) in img1_neighbors:
    #                     img2 = img_[:,:,i2-ps:i2+ps+1,j2-ps:j2+ps+1]

    #                     # feeding in the same image should result in a low number
    #                     # print(img1.shape, img2.shape)
    #                     dists.append(model(img1, img2))
    #                     dists_idx.append(([di1, dj1], [di2, dj2]))
    #                 else:
    #                     # dists.append(500)
    #                     pass
    #                     # dists_idx.append(([di1, dj1], [di2, dj2]))
    #         print(i1, j1)
    # end = time()
    # print(end-start)