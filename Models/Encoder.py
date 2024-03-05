import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, n_features, n_heads, n_hidden=64, dropout=0.1):
        """
        Args:
          n_features: Number of input and output features.
          n_heads: Number of attention heads in the Multi-Head Attention.
          n_hidden: Number of hidden units in the Feedforward (MLP) block.
          dropout: Dropout rate after the first layer of the MLP and in two places on the main path (before
                   combining the main path with a skip connection).
        """

        super(EncoderBlock, self).__init__()
        self.atten = nn.MultiheadAttention(n_features,n_heads)
        self.Drop1 = nn.Dropout(dropout)
        self.Norm1 = nn.LayerNorm(n_features)

        self.FeedFW = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features),
        )

        self.Drop2 = nn.Dropout(dropout)
        self.Norm2 = nn.LayerNorm(n_features)


    def forward(self, x):
        """
        Args:
          x of shape (max_seq_length, batch_size, n_features): Input sequences.
          mask of shape (max_seq_length, batch_size): BoolTensor indicating which elements of the input
              sequences should be ignored (True values correspond to ignored elements in x).

        Returns:
          z of shape (max_seq_length, batch_size, n_features): Encoded input sequences.

        Note: All intermediate signals should be of shape (max_seq_length, batch_size, n_features).
        """

        atten_out, atten_out_para = self.atten(x,x,x)
        h1 = self.Norm1(x + self.Drop1(atten_out))
        h2 = self.Norm2(h1 + self.Drop2(self.FeedFW(h1)))

        return h2
        raise NotImplementedError()