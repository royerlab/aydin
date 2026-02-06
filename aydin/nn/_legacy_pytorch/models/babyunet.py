"""BabyUnet - a lightweight 2-level UNet architecture.

Provides a minimal UNet with two encoder/decoder levels using
average pooling and bilinear upsampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from aydin.nn._legacy_pytorch.models.convblock import ConvBlock


class BabyUnet(nn.Module):
    """Lightweight 2-level UNet for image denoising.

    A minimal UNet with two encoder levels, average pooling for
    downsampling, and bilinear interpolation for upsampling.

    Parameters
    ----------
    n_channel_in : int
        Number of input channels.
    n_channel_out : int
        Number of output channels.
    """

    def __init__(self, n_channel_in=1, n_channel_out=1):
        super(BabyUnet, self).__init__()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.up1 = lambda x: F.interpolate(x, mode='bilinear', scale_factor=2)
        self.up2 = lambda x: F.interpolate(x, mode='bilinear', scale_factor=2)

        self.conv1 = ConvBlock(n_channel_in, 16)
        self.conv2 = ConvBlock(16, 32)

        self.conv3 = ConvBlock(32, 32)

        self.conv4 = ConvBlock(64, 32)
        self.conv5 = ConvBlock(48, 16)

        self.conv6 = nn.Conv2d(16, n_channel_out, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        x = self.pool1(c1)
        c2 = self.conv2(x)
        x = self.pool2(c2)
        x = self.conv3(x)

        x = self.up1(x)
        x = torch.cat([x, c2], 1)
        x = self.conv4(x)
        x = self.up2(x)
        x = torch.cat([x, c1], 1)
        x = self.conv5(x)
        x = self.conv6(x)
        return x
