from torch.utils.data import Dataset, DataLoader
import torch.functional as F
import torch.nn as nn
import torch

import h5py

import numpy as np
import os
import random
import matplotlib.pyplot as plt

class Dataset(Dataset):
    def __init__(self, train=True, split_size=0.95, patch_size=15, patches=True):  
        assert 0 < split_size < 1
        self.file = h5py.File('nyu_depth_v2_labeled.mat')
        self.train = train
        self.split_size = split_size
        self.shape = [480, 640, 4]
        self.ps = patch_size//2
        self.skip_len = patch_size+1
        self.run_patches = patches
    def __getitem__(self, item):
        assert self.__len__() > item
        if not self.train:
            item += int(len(self.file['images']) * self.split_size)
        if self.run_patches:
            return self.generate_ABC_pairs()

        img = self.file['images'][item]
        depth = self.file['depths'][item]
        # reshape
        img_ = np.empty([480, 640, 4])
        img_[:,:,0] = img[0,:,:].T / 255
        img_[:,:,1] = img[1,:,:].T / 255
        img_[:,:,2] = img[2,:,:].T / 255
        img_[:,:,3] = depth[:,:].T # already in range 0-1

        return img_.astype(np.float32)
    def __len__(self):
        ratio = self.split_size if self.train else (1-self.split_size)
        return int(len(self.file['images']) * ratio)
    def get_from_xy(self, x, y, img_):
        return img_[:,:,x-self.ps:x+self.ps+1,y-self.ps:y+self.ps+1]
    def choose_rand_patch(self):
        x = np.random.choice(range(self.ps, self.shape[0], self.skip_len))
        y = np.random.choice(range(self.ps, self.shape[1], self.skip_len))
        return x, y
    def choose_rand_neighbor(self, x, y):
        skip_len = self.skip_len
        xyc = [(x+skip_len, y), (x-skip_len, y), (x+skip_len, y+skip_len), (x-skip_len, y+skip_len),
               (x+skip_len, y-skip_len), (x-skip_len, y-skip_len), (x, y+skip_len), (x, y-skip_len)]
        xyc = filter(lambda x: self.ps<x[0]<self.shape[0]-self.ps and 
                                  self.ps<x[1]<self.shape[1]-self.ps, xyc)
        xyc = list(xyc)
        return random.choice(xyc)
    def generate_ABC_pairs(self, load_img = False):
        # Generate all patches A and B with their corresponding C value
        #------- Get an Image -------#
        item = np.random.choice(range(0, self.__len__()))
        img = self.file['images'][item]
        depth = self.file['depths'][item]
        # reshape
        img_ = np.empty([480, 640, 4])
        img_[:,:,0] = img[0,:,:].T / 255
        img_[:,:,1] = img[1,:,:].T / 255
        img_[:,:,2] = img[2,:,:].T / 255
        img_[:,:,3] = depth[:,:].T # already in range 0-1
        img=img_.copy()
        # Normalize
        for i in [3]:#range(self.shape[2]):
            img_[:,:,i] = (img_[:,:,i]-np.mean(img_[:,:,i])) / np.std(img_[:,:,i])
        #------ Create Patches ------#
        if np.random.rand() > .5: # make an example with C=1
            A_x, A_y = self.choose_rand_patch()
            B_x, B_y = self.choose_rand_neighbor(A_x, A_y)
            img[A_x-self.ps:A_x+self.ps+1, A_y-self.ps:A_y+self.ps+1,:]=0
            img[B_x-self.ps:B_x+self.ps+1, B_y-self.ps:B_y+self.ps+1,:]=0  
            if load_img:          
                return img, img_[A_x-self.ps:A_x+self.ps+1, A_y-self.ps:A_y+self.ps+1,:], img_[B_x-self.ps:B_x+self.ps+1, B_y-self.ps:B_y+self.ps+1,:], 1
            return img_[A_x-self.ps:A_x+self.ps+1, A_y-self.ps:A_y+self.ps+1,:], img_[B_x-self.ps:B_x+self.ps+1, B_y-self.ps:B_y+self.ps+1,:], 1
        else:                     # make an example with C=0
            A_x, A_y = self.choose_rand_patch()
            B_x, B_y = self.choose_rand_patch()
            while (B_x, B_y) == (A_x, A_y): # make sure A and B not the same here
                B_x, B_y = self.choose_rand_patch()
            img[A_x-self.ps:A_x+self.ps+1, A_y-self.ps:A_y+self.ps+1,:]=0
            img[B_x-self.ps:B_x+self.ps+1, B_y-self.ps:B_y+self.ps+1,:]=0
            if load_img:
                return img, img_[A_x-self.ps:A_x+self.ps+1, A_y-self.ps:A_y+self.ps+1,:], img_[B_x-self.ps:B_x+self.ps+1, B_y-self.ps:B_y+self.ps+1,:], 0   
            return img_[A_x-self.ps:A_x+self.ps+1, A_y-self.ps:A_y+self.ps+1,:], img_[B_x-self.ps:B_x+self.ps+1, B_y-self.ps:B_y+self.ps+1,:], 0


if __name__ == '__main__':
    data = Dataset()
    for i in range(10):
        img, a, b, c = data.generate_ABC_pairs()
        plt.figure()
        plt.imshow(img[:,:,(0, 1, 3)])
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(a[:,:,(0,1,3)])
        plt.title(str(c))
        plt.subplot(1, 2, 2)
        plt.imshow(b[:,:,(0,1,3)])
        plt.title(str(c))
        plt.show()
