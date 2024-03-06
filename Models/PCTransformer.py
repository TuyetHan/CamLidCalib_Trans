import torch
import torch.nn as nn
from Grid_Sampling import grid_sample
from KPConvBlock import KPConvSimpleBlock
from Encoder import EncoderBlock

class PC_Trs(nn.Module):
      def __init__(self, device, in_channels, conv_channels, conv_blocks,
                   mlp_blocks, mlp_feature, mlp_heads,
                   window_size, max_pc_voxel,
                   mlp_hidden=64, mlp_dropout=0.1,
                   prev_grid_size=0.04, sigma=1.0):
          """
          Args:
            in_channels: Channels of point cloud input (XYZ only: 3, XYZ + Colour: 6)
            conv_channels: out channels of KPConv
            conv_blocks: Number of KPConvBlock

            mlp_blocks: Number of EncoderBlock blocks.
            mlp_feature: Number of features to be used for word embedding and further in all layers of the encoder.
            mlp_heads:  Number of attention heads inside the EncoderBlock.
            mlp_hidden: Number of hidden units in the Feedforward block of EncoderBlock.
            mlp_dropout: Dropout level used in EncoderBlock.
          """
          super(PC_Trs, self).__init__()
          self.window_size = window_size
          self.max_pc_voxel = max_pc_voxel
          self.device = device

          self.KPBlocks = nn.ModuleList([
              KPConvSimpleBlock(in_channels=in_channels, out_channels=conv_channels,
                                prev_grid_size=prev_grid_size, sigma=sigma)
              for i in range(conv_blocks)])

          self.EncBlocks = nn.ModuleList(
              [EncoderBlock(n_features=mlp_feature, n_heads=mlp_heads, n_hidden=mlp_hidden, dropout=mlp_dropout)
              for i in range(mlp_blocks)])

          # TODO: 3 change to (conv_channels+3)
          self.weightmatrix = nn.Parameter(torch.randn(1, 3, mlp_feature, device = self.device))

      def forward(self, xyz, feature, batch_idx, neighbor_idx):
          """
          Args:
            xyx of shape (batch_size, number of point, 3): Point Cloud x,y,z value
            feature of shape(batch_size, number of point, D): D can be 1,3,...
            batch_idx of shape(batch_size*number of point, 1): index use to indicate point belong to which batch
            neighbor index ()

          Returns:
            z of shape (batch_size): Encoded output.

          Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
          """

          # KPConv Layer
          for i, layer in enumerate(self.KPBlocks):
              feature = layer(feature, xyz, batch = batch_idx, neighbor_idx = neighbor_idx)
          feature = feature.contiguous()

          # Partition pointcloud to non-overlapping window
          window_size = torch.tensor([self.window_size]*3).type_as(xyz).to(xyz.device)
          list_batch_cluster = grid_sample(xyz, batch_idx, window_size, start=None, max_pc_voxel = self.max_pc_voxel) # a list of shape(B,N,3)

          # Attention Encoder
          output = []
          for cluster in (list_batch_cluster):
            flatten_PC = torch.flatten(cluster, start_dim=2).permute(1,0,2)  #transpose
            embed = torch.matmul(flatten_PC.float() , self.weightmatrix)

            for layer in (self.EncBlocks):
                embed = layer(embed)

            output.append(embed)
          y = torch.cat(output, dim=0)

          return y