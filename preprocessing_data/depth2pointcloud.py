from open3d import *
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def point_cloud(depth):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    z = np.where(valid, depth / 256.0, np.nan)
    x = np.where(valid, z * (c) / 1, 0)
    y = np.where(valid, z * (r) / 1, 0)
    return np.dstack((x, y, z))

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
    img_[:,:,0] = img[0,:,:].T / 255
    img_[:,:,1] = img[1,:,:].T / 255
    img_[:,:,2] = img[2,:,:].T / 255
    img_[:,:,3] = depth[:,:].T # already in range 0-1

    print(img_.shape)
    # plt.figure()
    # plt.imshow(img_)
    # plt.show()
    i1 = Image((img_[:,:,:3]*255).astype(np.uint8))
    i2 = Image((img_[:,:,-1]).astype(np.float32))
    rgbd_image = create_rgbd_image_from_nyu_format(i1, i2)
    pcd = create_point_cloud_from_rgbd_image(rgbd_image, PinholeCameraIntrinsic(
            PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    draw_geometries([pcd])
    # pcd = PointCloud()
    # pcd.points = Vector3dVector(img_)
    
    # draw_geometries([pcd_load])
    # pcd = point_cloud(img_[::10,::10,-1])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # print(np.array(pcd).shape)
    # ax.scatter(*np.dsplit(pcd, 3), c=img_[::10,::10,0].reshape(-1)
    #                                 +img_[::10,::10,1].reshape(-1)
    #                                 +img_[::10,::10,2].reshape(-1))
    # plt.show()
    # iii = Image((img_[:,:,3]*255).astype(np.uint8))
    # draw_geometries([iii])

