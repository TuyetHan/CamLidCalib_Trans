import torch
import torch.nn as nn

from pytorch3d.ops import sample_farthest_points, ball_query
from pytorch3d.ops.utils import masked_gather
from Models.Encoder import EncoderBlock

def conv_block(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.ReLU()
    )

class PointNet(nn.Module):
    def __init__(self, num_points, feat_in_dim, 
                 n_sample, radius, conv_out_channels, 
                 kernel_size=[1, 1], padding=1, use_xyz = False):
        """
        Args:
            num_points: number of sample points in farthes point sampling
            feat_in_dim:  dimension of input feature
            n_sample: how many points in each local region  (list of int)
            radius:   radius of each local region           (list of float)
            
            
            conv_out_channels:   channel number of each convolution layer (list of int)
            kernel_size:  kernel size of each convolution layer
            padding:  padding of each convolution layer
            use_xyz:  whether to include xyz in feature
        """
        super(PointNet, self).__init__()
        self.num_points = num_points
        self.n_sample = n_sample
        self.radius = radius
        self.use_xyz = use_xyz
        self.feat_in_dim = feat_in_dim
        if use_xyz: feat_in_dim += 3

        self.cov_sizes = [feat_in_dim, *conv_out_channels]
        conv_blocks = [conv_block(in_f, out_f, kernel_size=kernel_size, padding=padding) 
                       for in_f, out_f in zip(self.cov_sizes, self.cov_sizes[1:])]
        self.ConvBlk = nn.Sequential(*conv_blocks)

        self.maxpool = nn.AdaptiveMaxPool2d((num_points, 1))
        

    def forward(self, pointcloud):
        """
        Args:
            pointcloud of shape (batch_size, number of point, 3+D): Point Cloud x,y,z value and features
                D is dimension of features
        Returns:
            out_feature of shape (batch_size, num_points, 3 + out_channel*len(radius list)): new xyz + new feature   
        """

        # Farthest point sampling - This will be new xyz
        position = pointcloud[:,:,:3]

        if self.use_xyz:
            features = pointcloud
        else:
            features = pointcloud[:,:,-self.feat_in_dim:] #take the last feat_in_dim collums
            
        sample_position, idx = sample_farthest_points(position, K = self.num_points)
        sample_features = masked_gather(features, idx)

        #Feature extraction
        feature_list = []
        for i in range (len(self.radius)):
            radius = self.radius[i]
            nsample = self.n_sample[i]
            _, _, feature_group = ball_query(sample_features, features, K = nsample, radius = radius, return_nn=True)   
            # feature_group of shape (batch_size, num_points, n_sample, dimension) #todo: Normalization here?

            # Perform 2D Convolution and pooling
            feature_group = self.ConvBlk(feature_group.permute(0,3,1,2))
            max_feature =  self.maxpool(feature_group).permute(0,2,3,1).squeeze(2)
            feature_list.append(max_feature)

        # Concatenate all features
        out_feature = torch.cat([sample_position]+ feature_list, dim=2) 
        return out_feature
    
class PointFeature_Trs(nn.Module):
    def __init__(self, device, num_points,feat_in_dim, 
                  n_sample, radius, conv_channels, 
                  mlp_blocks, mlp_feature, mlp_heads,

                  use_xyz=False,
                  mlp_hidden=64, mlp_dropout=0.1,
                  num_neighbor = 200, prev_grid_size=0.04, sigma=1.0):
        """
        Args:
          num_points: number of sample points in farthes point sampling (list of int)
          feat_in_dim:  dimension of input feature
          n_sample: how many points in each local region  (list of list int)
          radius:   radius of each local region           (list of list float)
          conv_channels:   channel number of each convolution layer (list of list int)
        """

        super(PointFeature_Trs, self).__init__()
        self.in_feat = [3*conv_channels[i][-1] for i in range(len(conv_channels))]
        self.in_feat.insert(0, feat_in_dim)
        pointnet_blocks = [PointNet( n_po, in_feat, n_sam, rad, conv_ch, use_xyz = use_xyz)
                      for n_po, in_feat, n_sam, rad, conv_ch in 
                      zip(num_points, self.in_feat, n_sample, radius, conv_channels)]
        self.PointNetBlock = nn.Sequential(*pointnet_blocks)
        
        self.Pooling = nn.AdaptiveAvgPool2d((None, mlp_feature))
        self.EncBlocks = nn.ModuleList(
              [EncoderBlock(n_features=mlp_feature, n_heads=mlp_heads, n_hidden=mlp_hidden, dropout=mlp_dropout)
              for i in range(mlp_blocks)])
        
    def forward(self, pointcloud):
        """
        Args:
            position of shape (batch_size, number of point, 3): Point Cloud x,y,z value
            feature of shape(batch_size, number of point, D): D can be 1,3,...
        Returns:
          out: (batch_size, num_classes)
        """
        feature = self.PointNetBlock(pointcloud)
        feature = self.Pooling(feature)
        for block in self.EncBlocks:
            feature = block(feature)

        return feature
