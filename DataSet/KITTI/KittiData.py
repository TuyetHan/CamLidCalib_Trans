import torch
from torch.utils.data import Dataset

import numpy as np
import cv2
import random

class KITTIData(Dataset):
    def __init__(self, files:list=[], feat_dim = 1, num_point = 62464, img_reshape = None):
        super().__init__()
        self.files = files
        self.angle_limit = 0.34906585039886592
        self.tr_limit = 0.4
        self.eps = 1e-8
        self.feat_dim = feat_dim
        self.num_point = num_point      #1024 
        self.img_reshape = img_reshape  #(187, 621)

    def __len__(self):
        return len(self.files)

    def __preprocImg__(self, imgPath):
        img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
        if self.img_reshape is not None: img = cv2.resize(img, self.img_reshape)
        img = img.transpose((2, 0, 1)) / 255.0
        return torch.tensor(img, dtype=torch.float32)

    def __preprocfeat__(self, pcdPath):
        # return self.feat_dim last collumn of point cloud
        feature = np.fromfile(pcdPath, dtype=np.float32).reshape((-1, 4))[:,-self.feat_dim:]
        feature = np.array(random.choices(feature, cum_weights=None,
                                              weights=np.ones(shape=(feature.shape[0], 1)), k=self.num_point))
        return torch.tensor(feature, dtype=torch.float32).squeeze()

    def getVelo2CamTransform(self, calibVelo2CamPath):
        calib_dict = {}
        with open(calibVelo2CamPath, 'r') as f_:
            f_.readline()
            for line in f_.readlines():
                t, d = line.split(':', 1)
                calib_dict[t] = np.fromstring(d, dtype=np.float32, sep=' ')

        velo_to_cam_R = calib_dict['R'].reshape(3,3)
        velo_to_cam_T = calib_dict['T'].reshape(3,1)
        velo_to_cam = np.vstack((np.hstack((velo_to_cam_R, velo_to_cam_T)), np.array([[0,0,0,1]])))
        return velo_to_cam

    def genCam2CamTransform(self, calibCam2CamPath):
        calib_dict = {}
        with open(calibCam2CamPath, 'r') as f_:
            f_.readline()
            for line in f_.readlines():
                t, d = line.split(':', 1)
                calib_dict[t] = np.fromstring(d, dtype=np.float32, sep=' ')

        R_rect_00 = np.identity(n=4)
        R_rect_00[:3, :3] = calib_dict['R_rect_00'].reshape(3,3)
        P_rect_02 = calib_dict['P_rect_02'].reshape(3, 4)
        fx = P_rect_02[0,0]
        fy = P_rect_02[1, 1]
        k = P_rect_02[:3, :3]
        cam_02_transform = np.identity(n=4)
        cam_02_transform[:3, 3] = P_rect_02[:, 3]
        cam_02_transform[0, 3] /= fx
        cam_02_transform[1, 3] /= fy
        rect2cam = cam_02_transform @ R_rect_00

        return k, rect2cam



    def __preprocPcd__(self, pcdPath, calibCam2CamPath, calibVelo2CamPath):
        velo_to_cam = self.getVelo2CamTransform(calibVelo2CamPath)
        K, rect2cam = self.genCam2CamTransform(calibCam2CamPath)
        lidar2img = rect2cam @ velo_to_cam
        points = np.fromfile(pcdPath, dtype=np.float32).reshape((-1, 4))[:,:3]
        ones_col = np.ones(shape=(points.shape[0],1))
        points = np.hstack((points,ones_col))
        # points = points[points[:, 0] > 0]

        omega_x = self.angle_limit*np.random.random_sample() - (self.angle_limit/2.0) + self.eps
        omega_y = self.angle_limit*np.random.random_sample() - (self.angle_limit/2.0) + self.eps
        omega_z = self.angle_limit*np.random.random_sample() - (self.angle_limit/2.0) + self.eps
        tr_x = self.tr_limit*np.random.random_sample() - (self.tr_limit/2.0)
        tr_y = self.tr_limit*np.random.random_sample() - (self.tr_limit/2.0)
        tr_z = self.tr_limit*np.random.random_sample() - (self.tr_limit/2.0)

        theta = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        omega_cross = np.array([0.0, -omega_z, omega_y, omega_z, 0.0, -omega_x, -omega_y, omega_x, 0.0]).reshape(3,3)

        A = np.sin(theta)/theta
        B = (1.0 - np.cos(theta))/(theta**2)

        R = np.identity(n=3) + A*omega_cross + B*np.matmul(omega_cross, omega_cross)

        T = np.array([tr_x, tr_y, tr_z]).reshape(3,1)

        random_transform = np.vstack((np.hstack((R, T)), np.array([[0.0, 0.0, 0.0, 1.0]])))


        points_in_cam_axis = np.matmul(lidar2img, points.T)
        transformed_points = np.matmul(random_transform, points_in_cam_axis)

        transformed_points = transformed_points.T
        transformed_points = np.array(random.choices(transformed_points, cum_weights=None,
                                                     weights=np.ones(shape=(transformed_points.shape[0], 1)), k=self.num_point))
        transformed_points = transformed_points.T

        return torch.tensor(transformed_points, dtype=torch.float32), torch.tensor(random_transform, dtype=torch.float32), \
                    torch.tensor(K, dtype=torch.float32)



    def __getitem__(self, idx):
        pcd_path = self.files[idx]['PointCloud_path']
        img_path = self.files[idx]['Image_path']
        calib_cam_to_cam_path= self.files[idx]['calib_cam_to_cam']
        calib_velo_to_cam_path= self.files[idx]['calib_velo_to_cam']

        img = self.__preprocImg__(img_path)
        feature = self.__preprocfeat__(pcd_path)
        depth, randTrans, K = self.__preprocPcd__(pcd_path, calib_cam_to_cam_path, calib_velo_to_cam_path)

        return {'Image': img, 'Depth': depth, 'Feature': feature, 'Transform': randTrans,
                'intrinsics': K}



