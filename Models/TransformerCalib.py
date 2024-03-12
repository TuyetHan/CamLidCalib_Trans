import torch
import torch.nn as nn
from VisionTransformer import Cam_ViT
from PCTransformer import PC_Trs
from BasicRNN import BasicRNN

class TransformerCalib(nn.Module):
      def __init__(self, device,
                   mlp_blocks, mlp_feature, mlp_heads,
                   cam_patch_dim, cam_conv_channels,
                   pc_feat_dim, pc_cluster_window_size, 
                   pc_conv_channels, pc_conv_blocks,

                   img_depth = 3, cam_patch_method = 1, cam_position_en_method = 1,
                   pc_pool_type =None, pc_out_size = None, pc_nei_quer_radius = 200,
                   num_neighbor = 200, prev_grid_size=0.04, sigma=1.0,
                   rt_channels = 384, rt_hidden_size = 256, rt_Dropout = 0.7,
                   mlp_hidden=64, mlp_dropout=0.1):
          """
          Args:
            mlp_blocks: Number of EncoderBlock blocks.
            mlp_feature: Number of features to be used for word embedding and further in all layers of the encoder.
            mlp_heads:  Number of attention heads inside the EncoderBlock.
            mlp_hidden: Number of hidden units in the Feedforward block of EncoderBlock.
            mlp_dropout: Dropout level used in EncoderBlock.
          """
          super(TransformerCalib, self).__init__()
          self.rt_channels = rt_channels
          self.feature = mlp_feature
          # Camera Feature Extract
          self.CameraTrans = Cam_ViT(device = device, img_depth = img_depth,
                                     patch_method = cam_patch_method, position_en_method = cam_position_en_method,
                                     patch_dim=cam_patch_dim, conv_channels = cam_conv_channels,

                                     mlp_blocks=mlp_blocks, mlp_feature = mlp_feature, mlp_heads=mlp_heads,
                                     mlp_hidden=mlp_hidden, mlp_dropout=mlp_dropout)

          # PointCloud Feature Extract
          self.PCloudTrans = PC_Trs(device = device, feat_dim = pc_feat_dim,
                                    KP_conv_channels = pc_conv_channels, KP_conv_blocks = pc_conv_blocks,

                                    ouput_size = pc_out_size, cluster_window_size = pc_cluster_window_size,
                                    pool_type = pc_pool_type, nei_quer_radius = pc_nei_quer_radius,

                                    num_neighbor = num_neighbor, prev_grid_size=prev_grid_size, sigma=sigma,
                                    mlp_blocks=mlp_blocks, mlp_feature = mlp_feature, mlp_heads=mlp_heads,
                                    mlp_hidden=mlp_hidden, mlp_dropout=mlp_dropout)

          # For Translation and Rotation Estimation
          self.avgpoolTran = nn.AdaptiveAvgPool2d(output_size=(1,1))
          self.bnTran = nn.BatchNorm2d(num_features=self.rt_channels)
          self.convTran = nn.Conv2d(in_channels=1, out_channels=self.rt_channels,
                                    kernel_size=(1, 1), stride=(1,1), bias=False)

          self.avgpoolRot = nn.AdaptiveAvgPool2d(output_size=(1,1))
          self.bnRot = nn.BatchNorm2d(num_features = self.rt_channels)
          self.convRot = nn.Conv2d(in_channels= 1, out_channels= self.rt_channels,
                                   kernel_size=(1, 1), stride=(1,1), bias=False)

          self.relu = nn.ReLU(inplace=True)
          self.dropout = nn.Dropout(p = rt_Dropout)

          # Output Estimation
          self.rotrnn = BasicRNN(self.rt_channels, rt_hidden_size, 3)
          self.tranrnn = BasicRNN(self.rt_channels, rt_hidden_size, 3)

      def forward(self, image, position, feature, hidden_rot, hidden_tran):
          """
          Args:
            image of shape (batch_size, img height, img width, img_depth): Camera Image
            position of shape (batch_size, number of point, 3): Point Cloud x,y,z value
            feature of shape(batch_size, number of point, D): D can be 1,3,...

          Returns:
            z of shape (batch_size, ouput_size): ouput_size is define in the constructor

          Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
          """
          # Feature Extract
          img_feat = self.CameraTrans(image)
          lidar_feat = self.PCloudTrans(position, feature)
          all_feat = torch.cat((img_feat, lidar_feat), dim = 1).reshape(position.size(0), 1, -1, self.feature)

          # Estimate Rotation and Translation
          # Q: Should I replace by Attention k = lidar, q,v = img? or concate?
          rot_features = self.avgpoolRot(self.dropout(self.relu(self.bnRot(self.convRot(all_feat)))))
          rot_features = rot_features.flatten(1)
          tran_features = self.avgpoolTran(self.dropout(self.relu(self.bnTran(self.convTran(all_feat)))))
          tran_features = tran_features.flatten(1)

          outRot, hidden_rot = self.rotrnn(rot_features, hidden_rot)
          outTran, hidden_tran = self.tranrnn(tran_features, hidden_tran)

          return torch.concat([outTran, outRot], dim=1), hidden_rot, hidden_tran