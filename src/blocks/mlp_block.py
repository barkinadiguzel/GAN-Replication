import torch.nn as nn
from src.layers.linear import LinearLayer

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super().__init__()
        self.linear = LinearLayer(in_dim, out_dim)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x
