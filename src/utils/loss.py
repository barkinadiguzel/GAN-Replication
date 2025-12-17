import torch
import torch.nn.functional as F

def discriminator_loss(real_pred, fake_pred):
    real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
    fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
    return real_loss + fake_loss

def generator_loss(fake_pred):
    return F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))
