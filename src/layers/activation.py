import torch.nn as nn

class ReLU(nn.Module):
    def forward(self, x):
        return nn.ReLU()(x)

class Sigmoid(nn.Module):
    def forward(self, x):
        return nn.Sigmoid()(x)
