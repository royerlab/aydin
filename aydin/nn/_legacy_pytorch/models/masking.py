import torch
import torch.nn as nn


class Masking(nn.Module):
    def __init__(self, module):
        super().__init__()

        self.module: nn.Module = module
        self.density = 0.1
        self.mask = None

    def forward(self, x):
        self.mask = torch.rand_like(x) < self.density
        x = (~self.mask) * x
        x = self.module(x)
        return x

    def get_mask(self):
        return self.mask

    def trainable_parameters(self):
        return self.module.trainable_parameters()
