import torch.nn as nn
from src.blocks.mlp_block import MLPBlock
from src.layers.activation import ReLU, Sigmoid

class Discriminator(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            MLPBlock(data_dim, hidden_dim, ReLU()),
            MLPBlock(hidden_dim, hidden_dim, ReLU()),
            MLPBlock(hidden_dim, 1, Sigmoid())
        )

    def forward(self, x):
        return self.net(x)
