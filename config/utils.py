from typing import Any
import torch
import torch.nn as nn

from torchmetrics import StructuralSimilarityIndexMeasure
import numpy as np
import open3d as o3d

def save_pcd(filename:str, points:np.ndarray):
    o3d.io.write_point_cloud(filename, 
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))

def genNormPointcloud(SampledPointcloudGrid):
    point_cloud = SampledPointcloudGrid.permute((0, 2, 1))
    norms = torch.linalg.norm(point_cloud, axis=2)
    norm_maxes, _ = norms.max(axis=1)
    norm_maxes = norm_maxes[..., None, None]
    point_cloud = point_cloud / norm_maxes
    point_limit = 128
    no_points = point_cloud.shape[1]
    no_partitions = no_points//point_limit
    indices = torch.arange(0, no_partitions*point_limit, dtype=torch.long, device=point_cloud.device)
    point_cloud = point_cloud.index_select(1, indices)
    point_cloud = point_cloud.reshape([-1, point_limit, no_partitions, 3])
    point_cloud = point_cloud.mean(dim=2)

    return point_cloud

class LossDepthMap():
    def __init__(self, *args, **kwargs) -> None:
        super(LossDepthMap, self).__init__(*args, **kwargs)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred:torch.Tensor, exp:torch.Tensor):
        pred = nn.functional.avg_pool2d(pred, kernel_size=4, stride=4)
        exp = nn.functional.avg_pool2d(exp, kernel_size=4, stride=4)
        loss = self.mse(pred, exp)
        loss = (loss.sum((-2, -1)) * 0.5).mean()
        return loss
    
    def __call__(self, pred:torch.Tensor, exp:torch.Tensor):
        return self.forward(pred, exp)


class LossSSIM():
    def __init__(self, device:torch.device) -> None:
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.ssim = self.ssim.to(device=device)
    
    def forward(self, pred:torch.Tensor, exp:torch.Tensor):
        pred_idx = pred != 0.0
        exp_idx = exp != 0.0
        pred[pred_idx] = 1.0 - pred[pred_idx]
        exp[exp_idx] =  1.0 - exp[exp_idx]
        return 1.0 - self.ssim(pred, exp)
    
    def __call__(self, pred:torch.Tensor, exp:torch.Tensor):
        return self.forward(pred, exp)

def genDepthImageFromSampledGrid(K_Matrix, SampledPointcloudGrid, imageShape):
    batch_size, _, img_height, img_width = imageShape

    points_2d = torch.bmm(K_Matrix, SampledPointcloudGrid)

    reprojected_img = torch.zeros(size=(batch_size, 1, img_height, img_width), device=K_Matrix.device, dtype=torch.float32)

    for bt in range(0, batch_size):
        Z_bt = points_2d[bt, 2,:]
        x_bt = points_2d[bt, 0,:]
        y_bt = points_2d[bt, 1,:]
        mask_Z = torch.where((Z_bt > 0))[0]

        Z_bt = Z_bt[mask_Z]
        if Z_bt.numel() != 0:
            x_bt = x_bt[mask_Z] / Z_bt
            y_bt = y_bt[mask_Z] / Z_bt

            mask = torch.where((x_bt < img_width) & (x_bt >= 0) &
                        (y_bt < img_height) & (y_bt >= 0))[0]
            
            Z_bt = Z_bt[mask]
            y_idx = y_bt[mask].to(dtype=torch.long)
            x_idx = x_bt[mask].to(dtype=torch.long)
            if Z_bt.numel() != 0:
                min_depth = Z_bt.min()
                max_depth = Z_bt.max()
                # reprojected_img[bt, 0, y_idx, x_idx] = (Z_bt - min_depth)/(max_depth - min_depth)
                nZ_bt = (Z_bt - min_depth)/(max_depth - min_depth)
                reprojected_img[bt, 0, y_idx, x_idx] = (nZ_bt * 2.0) - 1.0
            
    return reprojected_img

def genWarpedSamplingGrid(depth_info, transform):
    warped_sampling_grid = torch.bmm(transform, depth_info)[:, :-1, :]

    return warped_sampling_grid


def gen_point_cloud_img(depth_info, predicted_transform, expected_transform, intrinsics, img_shape):
    predicted_warped_sampling_grid = genWarpedSamplingGrid(depth_info, predicted_transform)
    expected_warped_sampling_grid = genWarpedSamplingGrid(depth_info, torch.linalg.inv(expected_transform))

    predicted_point_cloud = genNormPointcloud(predicted_warped_sampling_grid)
    expected_point_cloud = genNormPointcloud(expected_warped_sampling_grid)

    # predicted_reprojected_img = genDepthImageFromSampledGrid(intrinsics, predicted_warped_sampling_grid, img_shape)
    # expected_reprojected_img = genDepthImageFromSampledGrid(intrinsics, expected_warped_sampling_grid, img_shape)

    return {'pcd':{'pred':predicted_point_cloud, 'exp':expected_point_cloud}, 
            'depthimg':{'pred':None, 'exp':None},
            }


def genTransformMat(out:torch.tensor=None):
    batch_size, _ = out.shape
    device = out.device
    u = out[:, :3]
    omega = out[:, 3:]

    theta = torch.sqrt(omega[:, 0]*omega[:, 0] + omega[:, 1]*omega[:, 1] + omega[:, 2]*omega[:, 2])
    zeros = torch.zeros((batch_size), device=device)
    omega_cross = torch.stack([zeros, -omega[:, 2], omega[:, 1], omega[:, 2], zeros, -omega[:, 0], -omega[:, 1], omega[:, 0], zeros])
    omega_cross = omega_cross.transpose(1,0)
    omega_cross = omega_cross.reshape([-1, 3, 3])

    A = (torch.sin(theta) / theta).to(device=device)
    B = ((1.0 - torch.cos(theta))/(torch.pow(theta, 2))).to(device=device)
    C = ((1.0 - A)/(torch.pow(theta, 2))).to(device=device)

    omega_cross_square = torch.bmm(omega_cross, omega_cross)

    R = torch.eye(3,3, device=device).repeat(batch_size, 1, 1) + A[..., None, None] * omega_cross + B[..., None, None] * omega_cross_square
    V = torch.eye(3,3, device=device).repeat(batch_size, 1, 1) + B[..., None, None] * omega_cross + C[..., None, None] * omega_cross_square
    Vu = torch.bmm(V, u[..., None])
    T = torch.concat([R, Vu], 2)
    T = torch.concat([T, torch.tensor([0., 0., 0., 1.], dtype=torch.float32, device=device).repeat(batch_size, 1, 1)], 1)
    T = torch.where(torch.isnan(T), torch.eye(n=4, device=device), T)

    return T

def transformDepth(depth_info, intrinsics, transform, img_shape):
    warped_sampling_grid = genWarpedSamplingGrid(depth_info, transform)
    depth_map = genDepthImageFromSampledGrid(intrinsics, warped_sampling_grid, img_shape)
    return depth_map

def rotationLoss(rotPred, rotExp):
    batch_size, _, _ = rotPred.shape
    Rot = torch.bmm(rotPred.permute(0, 2, 1), rotExp)
    Rot_T = Rot.permute(0, 2, 1)
    TraceRot = torch.einsum('bii->b', Rot)
    theta = torch.arccos((TraceRot - 1)*0.5)
    sin_theta = torch.sin(theta)
    norm_log = torch.zeros(size=(batch_size,))
    log_r = theta[..., None, None]*(Rot - Rot_T) 

    for bt in range(batch_size):
        sin_theta_bt = sin_theta[bt]
        if not sin_theta_bt == 0.0:
            log_r[bt] /= (2*sin_theta_bt)
        
        norm_log[bt] = torch.linalg.norm(log_r[bt])

    return norm_log.mean()


def getYawPitchRoll(RotMat):
    batch_size, _, _ = RotMat.shape
    pitch = torch.arctan2(-RotMat[:, 2, 0], torch.sqrt(RotMat[:, 0, 0]*RotMat[:, 0, 0] + RotMat[:, 1, 0]*RotMat[:, 1, 0]))
    yaw = torch.zeros(size=(batch_size,), dtype=torch.float32)
    roll = torch.zeros(size=(batch_size,), dtype=torch.float32)

    for bt in range(batch_size):
        if torch.isclose(pitch[bt], torch.tensor([-(torch.pi/2)], dtype=torch.float32)):
            yaw[bt] = torch.atan2(-RotMat[bt, 1, 2], -RotMat[bt, 0, 2])
        elif torch.isclose(pitch[bt], torch.tensor([torch.pi/2], dtype=torch.float32)):
            yaw[bt] = torch.atan2(RotMat[bt, 1, 2], RotMat[bt, 0, 2])
        else:
            yaw[bt] = torch.atan2(RotMat[bt, 1, 0], RotMat[bt, 0, 0])
            roll[bt] = torch.atan2(RotMat[bt, 2, 1], RotMat[bt, 2, 2])

    return torch.vstack([yaw, pitch, roll]) 


def anglesLoss(rotPred, rotExp):
    Rot = torch.bmm(rotPred.permute(0, 2, 1), rotExp)
    out = getYawPitchRoll(Rot)

    out = torch.abs(out).mean(dim=1)

    return torch.rad2deg(out)


def translaitionLoss(tranPred, tranExp):
     return torch.abs(tranPred - tranExp).mean(dim=0)