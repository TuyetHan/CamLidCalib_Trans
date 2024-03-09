import torch
import torch.nn as nn
import torch_points_kernels as tp

from Grid_Sampling import grid_sample
from KPConvBlock import KPConvSimpleBlock
from Encoder import EncoderBlock

class PC_Trs(nn.Module):
      def __init__(self, device,
                   in_channels, KP_conv_channels, KP_conv_blocks,
                   mlp_blocks, mlp_feature, mlp_heads,

                   ouput_size, cluster_window_size, nei_quer_radius, num_neighbor = 200,
                   pool_type ='max', size_a_pool = (5,1),

                   mlp_hidden=64, mlp_dropout=0.1,
                   prev_grid_size=0.04, sigma=1.0):
          """
          Args:
            in_channels: Channels of point cloud input (XYZ only: 3, XYZ + Colour: 6)
            KP_conv_channels: out channels of KPConv
            KP_conv_blocks: Number of KPConvBlock

            ouput_size: Size of output feature tensor
            cluster_window_size: Size of cluster window
            nei_quer_radius: Radius of neighbor query
            num_neighbor: Number of neighbor to query
            pool_type: Type of pooling (max or avg)
            size_a_pool: Size after pooling

            mlp_blocks: Number of EncoderBlock blocks.
            mlp_feature: Number of features to be used for word embedding and further in all layers of the encoder.
            mlp_heads:  Number of attention heads inside the EncoderBlock.
            mlp_hidden: Number of hidden units in the Feedforward block of EncoderBlock.
            mlp_dropout: Dropout level used in EncoderBlock.
          """
          super(PC_Trs, self).__init__()
          self.cluster_window_size = cluster_window_size
          self.radius = nei_quer_radius
          self.num_neighbor = num_neighbor
          self.size_a_pool = size_a_pool
          self.device = device

          self.KPBlocks = nn.ModuleList([
              KPConvSimpleBlock(in_channels=in_channels, out_channels=KP_conv_channels,
                                prev_grid_size=prev_grid_size, sigma=sigma)
              for i in range(KP_conv_blocks)])

          if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(size_a_pool)
            self.outpool = nn.AdaptiveAvgPool2d(ouput_size)
          else:
            self.pool = nn.AdaptiveMaxPool2d(size_a_pool)
            self.outpool = nn.AdaptiveMaxPool2d(ouput_size)

          self.EncBlocks = nn.ModuleList(
              [EncoderBlock(n_features=mlp_feature, n_heads=mlp_heads, n_hidden=mlp_hidden, dropout=mlp_dropout)
              for i in range(mlp_blocks)])

          self.weightmatrix = nn.Parameter(torch.randn(1, (KP_conv_channels + 3), mlp_feature, device = self.device))

      def forward(self, position, feature):
          """
          Args:
            pointcloud of shape (batch_size, number of point, N): Point Cloud (x,y,z) and feature(RGB)/(ref)
            position of shape (batch_size, number of point, 3): Point Cloud x,y,z value
            feature of shape(batch_size, number of point, D): D can be 1,3,...
            batch_idx of shape(batch_size*number of point, 1): index use to indicate point belong to which batch
            neighbor index ()

          Returns:
            z of shape (batch_size): Encoded output.

          Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
          """
          # Calculate Input Parameter
          batch_size = position.size(0)
          num_point  = position.size(1)
          window = torch.tensor([self.cluster_window_size]*3).type_as(position).to(self.device)

          # Change data location to GPU
          position = position.contiguous().view(-1, 3).to(self.device)
          feature  = feature.contiguous().view(-1, 3).to(self.device)

          # Batch_idx and Neighbor idx for Clustering and Convolution
          batch_idx = torch.cat([torch.tensor([ii]*o) for ii, o in enumerate([num_point]*batch_size)], 0).long().view(-1, 1).to(self.device)
          neighbor_idx = tp.ball_query(radius = self.radius, nsample = self.num_neighbor,
                                        x = position, y = position,
                                        mode="partial_dense",
                                        batch_x=batch_idx.squeeze(), batch_y=batch_idx.squeeze())[0].to(self.device)
          # KPConv Layer
          for i, layer in enumerate(self.KPBlocks):
              feature = layer(feature, position, batch = batch_idx, neighbor_idx = neighbor_idx)
          kpfeature = feature.contiguous()

          # Partition pointcloud to non-overlapping window 2D list [B, Clus] of tensor point [points, position+feature]
          all_clusters_points, max_n_clus = grid_sample(position, kpfeature, batch_idx, window, start=None)

          # Pooling all points in one cluster
          locpool_features = []
          for batch in all_clusters_points:
            locpool_feat = [self.pool(cluster_point.unsqueeze(0)).squeeze() for cluster_point in batch]
            locpool_features.append(locpool_feat)

          # Perform Attention
          output = []
          for sublist in range(max_n_clus):
            batch_feat = [(sublist[i] if i < len(sublist)
                                      else torch.zeros(self.size_a_pool).to(self.device)) for sublist in locpool_features]
            batch_feat = torch.stack(batch_feat).permute(1,0,2)
            embed = torch.matmul(batch_feat.float() , self.weightmatrix)

            for layer in (self.EncBlocks):
                embed = layer(embed)
            output.append(embed)

          # Perform pooling on output to obtain fix-sized
          output = self.outpool(torch.stack(output).permute(1,2,3,0))
          return output