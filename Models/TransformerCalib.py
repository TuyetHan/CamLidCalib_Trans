import torch
import torch.nn as nn
from .VisionTransformer import Cam_ViT
from .PCTransformer import PC_Trs
from .PFeatTransformer import PointFeature_Trs
from .BasicRNN import BasicRNN
from .Encoder import EncoderBlock

class TransformerCalib(nn.Module):
    def __init__(self, device, args):
        """
        Args:
          img_depth:  (int)    Number of channels in the input image. (RGB = 3)
          pc_feat_dim:(int)    Feature dimension of the point cloud.

          mlp_blocks: (int)     Number of EncoderBlock blocks.
          mlp_feature:(int)     Number of features in all layers of the EncoderBlock.
          mlp_heads:  (int)     Number of attention heads inside the EncoderBlock.
          (opt) mlp_hidden: (int)    Number of hidden units in the Feedforward.
          (opt) mlp_dropout:(flo)    Dropout level used in EncoderBlock.

          rt_channels:   (int)  Number of channels of Rotation,Translation Gen Network.
          rt_Dropout:    (flo)  Dropout level.
          rt_hidden_size:(int)  Number of hidden units.

          cam_patch_method:       (1,2,3) Method to extract patches from the image.
          cam_patch_dim:          (int)   Size of patch.
          cam_position_en_method: (1,2)   Method to encode the position of the patches.
          cam_conv_channels:      (int)   Number of channels in the convolutional.

          pc_conv_channels        (int)             Number of channels of KPConv.
          pc_conv_blocks          (int)             Number of KPConv blocks.
          (opt) pc_pool_type    (None, max, avg)    Type of pooling layer. 
          (opt) pc_out_size     (None, [N,M])       Output size of the point cloud feature.
          pc_cluster_window_size  (float)           Window size for clustering.
          pc_nei_quer_radius      (float)           Radius for querying neighbors.    tp.ball_query
          (opt) num_neighbor      (int)             Number of neighbors to query.     tp.ball_query
          (opt) prev_grid_size    (float)           Size of the previous grid.        KPConv
          (opt) sigma             (float)           Sigma value for the kernel.       KPConv
        """
        super(TransformerCalib, self).__init__()
        self.rt_hidden_size = args.rt_hidden_size
        self.rt_channels = args.rt_channels
        self.mlp_feature = args.mlp_feature
        self.pc_arch = args.pc_arch
        self.method = 2

        # Camera Feature Extract
        self.CameraTrans = Cam_ViT(device = device, img_depth = args.img_depth,
            patch_method = args.cam_patch_method, position_en_method = args.cam_position_en_method,
            patch_dim=args.cam_patch_dim, conv_channels = args.cam_conv_channels,

            mlp_blocks=args.mlp_blocks, mlp_feature = args.mlp_feature, mlp_heads=args.mlp_heads,
            mlp_hidden=args.mlp_hidden, mlp_dropout=args.mlp_dropout)

        # PointCloud Feature Extract
        if args.pc_arch == "PCTrans":
          self.PCloudTrans = PC_Trs(device = device, feat_dim = args.pc_feat_dim,
              KP_conv_channels = args.pc_conv_channels, KP_conv_blocks = args.pc_conv_blocks,

              ouput_size = args.pc_out_size, cluster_window_size = args.pc_cluster_window_size,
              pool_type = args.pc_pool_type, nei_quer_radius = args.pc_nei_quer_radius,

              num_neighbor = args.num_neighbor, prev_grid_size=args.prev_grid_size, sigma=args.sigma,
              mlp_blocks=args.mlp_blocks, mlp_feature = args.mlp_feature, mlp_heads=args.mlp_heads,
              mlp_hidden=args.mlp_hidden, mlp_dropout=args.mlp_dropout)
        elif args.pc_arch == "PFTrans":
          self.PCloudTrans = PointFeature_Trs(device = device, num_points=args.n_furthest_sam_points, 
              feat_in_dim=args.pc_feat_dim, n_sample = args.n_neighbor_sample, 
              radius=args.radius, conv_channels=args.pf_conv_channels,
              mlp_blocks=args.mlp_blocks, mlp_feature = args.mlp_feature, mlp_heads=args.mlp_heads,
              mlp_hidden=args.mlp_hidden, mlp_dropout=args.mlp_dropout)

        # For Translation and Rotation Estimation
        self.allFeatPool = nn.AdaptiveAvgPool2d(output_size=(self.rt_hidden_size,self.rt_channels))
        self.EncBlocks = nn.ModuleList(
            [EncoderBlock(n_features=self.rt_channels, n_heads=args.mlp_heads, 
                          n_hidden=self.rt_hidden_size, dropout=args.mlp_dropout)
                          for i in range(args.mlp_blocks)])

        self.avgpoolRot  = nn.AdaptiveAvgPool1d(output_size = 1)
        self.avgpoolTran = nn.AdaptiveAvgPool2d(output_size = 1)

        # Output Estimation
        self.rotrnn = BasicRNN(self.rt_channels, args.rt_hidden_size, 3)
        self.tranrnn = BasicRNN(self.rt_channels, args.rt_hidden_size, 3)

    def forward(self, image, position, feature, img_depth = None):
        """
        Args:
          image of shape (batch_size, img height, img width, img_depth): Camera Image
          position of shape (batch_size, number of point, 3): Point Cloud x,y,z value
          feature of shape(batch_size, number of point, D): D can be 1,3,...
          img_depth of shape (batch_size, img height, img width, 1): Depth Image

        Returns:
          z of shape (batch_size, ouput_size): ouput_size is define in the constructor

        Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
        """
        # Feature Extract
        if self.pc_arch == "ImgwDepth":
          image = torch.cat((image, img_depth), dim = 1)
        img_feat = self.CameraTrans(image)

        if self.pc_arch == "PCTrans":
          lidar_feat = self.PCloudTrans(position, feature)
        elif self.pc_arch == "PFTrans":
          point_wfeat = torch.cat((position, feature.unsqueeze(-1)), dim = 2)
          lidar_feat = self.PCloudTrans(point_wfeat)  

        # Estimate Rotation and Translation
        if self.pc_arch == "ImgwDepth":
          all_feat = img_feat
        else:
          all_feat = torch.cat((img_feat, lidar_feat), dim = 1).reshape(position.size(0), -1, self.mlp_feature)

        all_feat = self.allFeatPool(all_feat)
        for f in self.EncBlocks:
          all_feat = f(all_feat)

        rot_features  = self.avgpoolRot(all_feat.permute(0,2,1))
        tran_features = self.avgpoolRot(all_feat.permute(0,2,1))

        outRot  = self.rotrnn(rot_features.flatten(1))
        outTran = self.tranrnn(tran_features.flatten(1))

        return torch.concat([outTran, outRot], dim=1)