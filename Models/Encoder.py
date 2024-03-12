import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, n_features, n_heads, n_hidden=64, dropout=0.1, batch_first = True):
        """
        Args:
          n_features: Number of input and output features.
          n_heads: Number of attention heads in the Multi-Head Attention.
          n_hidden: Number of hidden units in the Feedforward (MLP) block.
          dropout: Dropout rate after the first layer of the MLP and in two places on the main path (before
                   combining the main path with a skip connection).
          batch_first: input will be (batch, seq, feature) instead of (seg, batch, feature)
        """

        super(EncoderBlock, self).__init__()
        self.atten = nn.MultiheadAttention(n_features,n_heads, batch_first = batch_first)
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


    def forward(self, x, attn_mask = None):
        """
        Args:
          x of shape (batch_size, max_seq_length, n_features): Input sequences.
          attn_mask of shape (batch_size*n_heads, max_seq_length, max_seq_length): 
              BoolTensor indicating which elements of the input sequences should be caculate
              2D matrix of (points, points) for each head, batch. Usefor PC Transformer

        Returns:
          z of shape (batch_size, max_seq_length, n_features): Encoded input sequences.
        """
        if attn_mask == None:
          atten_out, atten_out_para = self.atten(x,x,x)
        else:
          atten_out, atten_out_para = self.atten(x,x,x, attn_mask = attn_mask)
        h1 = self.Norm1(x + self.Drop1(atten_out))
        h2 = self.Norm2(h1 + self.Drop2(self.FeedFW(h1)))

        return h2
        raise NotImplementedError()