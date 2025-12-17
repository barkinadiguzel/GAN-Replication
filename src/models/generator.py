import torch.nn as nn
from src.blocks.mlp_block import MLPBlock
from src.layers.activation import ReLU

class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, data_dim):
        super().__init__()

        self.net = nn.Sequential(
            MLPBlock(z_dim, hidden_dim, ReLU()),
            MLPBlock(hidden_dim, hidden_dim, ReLU()),
            MLPBlock(hidden_dim, data_dim, None)
        )

    def forward(self, z):
        return self.net(z)
