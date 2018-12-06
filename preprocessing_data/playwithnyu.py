from open3d import *
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path_to_depth = 'nyu_depth_v2_labeled.mat'

def slices(img):
    patches = []
    for i in range(7, img.shape[0], 16):
        for j in range(7, img.shape[1], 16):
            patches.append(img[i-7:i+8,j-7:j+8,:])
            img[i-7:i+7,j-7:j+7,1] += (i/512+j/512)/2
    return patches
def get_rand_slice(): return None
# read mat file
f = h5py.File(path_to_depth)

batch_size=len(f['depths'])

imgs = np.zeros((batch_size,480,640,4))
# gt = np.zeros((batch_size,480,640,1))    
for i in range(len(f['depths'])):

    # read 0-th image. original format is [3 x 640 x 480], uint8
    img = f['images'][i]
    depth = f['depths'][i]

    # reshape
    img_ = np.empty([480, 640, 4])
    img_[:,:,0] = img[0,:,:].T / 255
    img_[:,:,1] = img[1,:,:].T / 255
    img_[:,:,2] = img[2,:,:].T / 255
    depth[:,:]  = depth[:,:] / np.max(depth[:,:])
    img_[:,:,3] = depth[:,:].T  # already in range 0-1

    print(img_.shape)
    print(np.min(img_[:,:,0]), np.max(img_[:,:,0]))
    print(np.min(img_[:,:,3]), np.max(img_[:,:,3]))
    plt.figure()
    plt.imshow(img_[:,:,:-1])
    plt.show()

    patches = slices(img_)

    plt.figure()
    plt.imshow(img_[:,:,:-1])
    plt.show()

    for i in range(10):
        print(patches[i].shape)
        plt.imshow(patches[np.random.choice(len(patches))][:,:,-1])
        plt.show()

    break
   