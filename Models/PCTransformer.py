import torch
import torch.nn as nn
import torch_points_kernels as tp

from .Grid_Sampling import grid_sample
from .KPConvBlock import KPConvSimpleBlock
from .Encoder import EncoderBlock

class PC_Trs(nn.Module):
      def __init__(self, device, feat_dim,
                   KP_conv_channels, KP_conv_blocks,
                   mlp_blocks, mlp_feature, mlp_heads,

                   cluster_window_size, nei_quer_radius,
                   pool_type = None, ouput_size = None,

                   mlp_hidden=64, mlp_dropout=0.1,
                   num_neighbor = 200, prev_grid_size=0.04, sigma=1.0):
          """
          Args:
            feat_dim: input feature dimension
            KP_conv_channels: out channels of KPConv
            KP_conv_blocks: Number of KPConvBlock

            ouput_size: Output size of the model have 2 values (N, M) without batch
            cluster_window_size: Size of the window for clustering
            nei_quer_radius: Radius for querying neighbors
            pool_type: Type of pooling to be used for the output of the model (max or avg)

            mlp_blocks: Number of EncoderBlock blocks.
            mlp_feature: Number of features to be used for word embedding and further in all layers of the encoder.
            mlp_heads:  Number of attention heads inside the EncoderBlock.
            mlp_hidden: Number of hidden units in the Feedforward block of EncoderBlock.
            mlp_dropout: Dropout level used in EncoderBlock.
          """
          super(PC_Trs, self).__init__()
          self.device = device
          self.feat_dim = feat_dim

          self.cluster_window_size = cluster_window_size
          self.radius = nei_quer_radius
          self.num_neighbor = num_neighbor
          
          self.mlp_heads = mlp_heads
          self.KP_conv_channels = KP_conv_channels
          self.pool_type = pool_type

          self.KPBlocks = nn.ModuleList([
              KPConvSimpleBlock(in_channels=feat_dim if i == 0 else KP_conv_channels, 
                  out_channels=KP_conv_channels, prev_grid_size=prev_grid_size, sigma=sigma)
              for i in range(KP_conv_blocks)])

          if pool_type == 'avg':
            assert ouput_size is not None
            self.outpool = nn.AdaptiveAvgPool2d(ouput_size)
          elif pool_type == 'max':
            assert ouput_size is not None
            self.outpool = nn.AdaptiveMaxPool2d(ouput_size)

          self.EncBlocks = nn.ModuleList(
              [EncoderBlock(n_features=mlp_feature, n_heads=mlp_heads, n_hidden=mlp_hidden, dropout=mlp_dropout)
              for i in range(mlp_blocks)])

          self.weightmatrix = nn.Parameter(torch.randn(1, (KP_conv_channels + 3), mlp_feature, device = self.device))

      def forward(self, position, feature):
          """
          Args:
            position of shape (batch_size, number of point, 3): Point Cloud x,y,z value
            feature of shape(batch_size, number of point, D): D can be 1,3,...

          Returns:
            z of shape (batch_size, ouput_size): ouput_size is define in the constructor

          Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
          """
          # Calculate Input Parameter
          batch_size = position.size(0)
          num_point  = position.size(1)
          window = torch.tensor([self.cluster_window_size]*3).type_as(position)

          # Change data location to GPU
          position_flat = position.contiguous().view(-1, 3)
          feature  = feature.view(-1, self.feat_dim)

          # Batch_idx and Neighbor idx for Clustering and Convolution
          batch_idx = torch.cat([torch.tensor([ii]*o) for ii, o in enumerate([num_point]*batch_size)], 0).long().view(-1, 1).to(self.device)
          neighbor_idx = tp.ball_query(radius = self.radius, nsample = self.num_neighbor,
                                        x = position_flat, y = position_flat,
                                        mode="partial_dense",
                                        batch_x=batch_idx.squeeze(), batch_y=batch_idx.squeeze())[0]

          # KPConv Layer
          for i, layer in enumerate(self.KPBlocks):
              feature = layer(feature, position_flat, batch = batch_idx , neighbor_idx = neighbor_idx)
          feature = feature.view(batch_size, -1, (self.KP_conv_channels))

          # Partition pointcloud to non-overlapping window - return a mask of size (Batch, n_point, n_point)
          cluster_mask = grid_sample(position_flat, batch_idx, window, start=None)
          cluster_mask = torch.repeat_interleave(cluster_mask, self.mlp_heads, dim = 0)

          # Linear Projection and Multihead Attention
          point_wfeature = torch.cat((position, feature), dim=2)
          embedding = torch.matmul(point_wfeature.float(), self.weightmatrix)
          for layer in (self.EncBlocks):
              embedding = layer(embedding, cluster_mask)

          # Perform pooling on output
          if self.pool_type is not None:
              embedding = self.outpool(embedding)

          return embedding