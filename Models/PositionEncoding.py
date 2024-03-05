import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """This implementation is the same as in the Annotated transformer blog post
        See https://nlp.seas.harvard.edu/2018/04/03/attention.html for more detail.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        assert (d_model % 2) == 0, 'd_model should be an even number.'
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PositionalEncoding_Conv(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2040):
        super(PositionalEncoding_Conv, self).__init__()

        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        identity = input.transpose(1, 2)

        out = self.conv1(identity)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + identity
        out = out.transpose(1, 2)
        return self.dropout(out)