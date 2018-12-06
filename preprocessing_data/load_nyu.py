import h5py
import numpy as np
import matplotlib.pyplot as plt

path_to_depth = 'nyu_depth_v2_labeled.mat'

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
    img_[:,:,0] = img[0,:,:].T
    img_[:,:,1] = img[1,:,:].T
    img_[:,:,2] = img[2,:,:].T
    img_[:,:,3] = depth[:,:].T

    imgs[i,:,:,:] = img_ 
    if i % 100 == 0:
    	print(i)
# return pred, gt
print(len(f['depths']))
print(imgs.shape)
np.savez_compressed("imgs", imgs)
# pred = pred.astype(int)
# # gt = gt.astype(int)

# for i in range(3):
# 	plt.figure()
# 	plt.imshow(gt[i,:,:,0])
# plt.show()