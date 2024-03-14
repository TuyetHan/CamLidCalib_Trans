import torch
import torch.nn as nn
import numpy as np
from Models.Encoder import EncoderBlock
from Models.PositionEncoding import PositionalEncoding, PositionalEncoding_Conv

class Cam_ViT(nn.Module):
    def __init__(self, device, img_depth, patch_method, position_en_method, patch_dim,
                 mlp_blocks, mlp_feature, mlp_heads,
                 mlp_hidden=64, mlp_dropout=0.1, conv_channels = None):
        """
        Args:
          img_depth: Depth of image (Black white: 1, Colour: 3)
          patch_method: Method to create image embedding
            1 - Split image with size = patch_dim and flatten it.
            2 - Conv2D with kernel_size = stride (No overlapping - have padding)
            3 - Conv2D with kernal_size != stride (Overlapping - no padding)
          patch_dim: H and W dimension of created patches.
          conv_channels: Number of Channels of Conv2D

          mlp_blocks: Number of EncoderBlock blocks.
          mlp_feature: Number of features to be used for word embedding and further in all layers of the encoder.
          mlp_heads:  Number of attention heads inside the EncoderBlock.
          mlp_hidden: Number of hidden units in the Feedforward block of EncoderBlock.
          mlp_dropout: Dropout level used in EncoderBlock.
        """

        super(Cam_ViT, self).__init__()
        self.patch_method = patch_method
        self.PatchDim = patch_dim
        self.img_depth = img_depth
        self.device = device

        # Create layers for patching image
        if self.patch_method == 1:
          self.Patch = nn.Unfold(kernel_size = (self.PatchDim, self.PatchDim), stride = self.PatchDim)
          self.weightmatrix = nn.Parameter(torch.randn(1, patch_dim * patch_dim * img_depth, mlp_feature, device = self.device))
        elif  self.patch_method == 2:
          # TODO: Consider padding here
          assert conv_channels is not None
          self.PatchConv = nn.Conv2d(img_depth, conv_channels, kernel_size = patch_dim, stride = patch_dim)
          self.weightmatrix = nn.Parameter(torch.randn(1, conv_channels, mlp_feature, device = self.device))
        elif self.patch_method == 3:
          assert conv_channels is not None
          self.PatchConv = nn.Conv2d(img_depth, conv_channels, kernel_size=patch_dim)
          self.weightmatrix = nn.Parameter(torch.randn(1, conv_channels, mlp_feature, device = self.device))

        # Positional Encoding here
        if position_en_method == 1:
          self.PosEnc = PositionalEncoding(mlp_feature)
        elif position_en_method == 2:
          self.PosEnc = PositionalEncoding_Conv(mlp_feature)

        # Transformer Encoder Architecture
        self.EncBlocks = nn.ModuleList(
            [EncoderBlock(n_features=mlp_feature, n_heads=mlp_heads, n_hidden=mlp_hidden, dropout=mlp_dropout)
             for i in range(mlp_blocks)])

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, img_depth, img height, img width): Input Image

        Returns:
          z of shape (batch_size, N, mlp_feature): Encoded output. N depends on patch method.
        """

        batch_size = x.shape[0]
        if  self.patch_method == 1:
          flatten_patches = nn.functional.unfold(x, (self.PatchDim, self.PatchDim), stride = self.PatchDim)

        elif self.patch_method == 2:
          patches = self.PatchConv(torch.Tensor(x))
          flatten_patches = torch.flatten(patches, start_dim=2)

        elif self.patch_method == 3:
          patches = self.PatchConv(torch.Tensor(x))
          flatten_patches = torch.flatten(patches, start_dim=2)
          # TODO: size off flatten patches very big. Pooling?

        # Linear Projection
        patch_embeddings = torch.matmul(torch.Tensor(flatten_patches.permute(0, 2, 1)) , self.weightmatrix)
        # Positional embeddings
        embedding = self.PosEnc(patch_embeddings)
        
        # Pass input through encoder transformer.
        for f in self.EncBlocks:
            embedding = f(embedding)

        return(embedding)