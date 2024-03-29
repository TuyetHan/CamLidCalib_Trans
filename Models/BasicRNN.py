import torch
import torch.nn as nn

class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicRNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size , hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        return output