"""Ronneberger UNet implementation adapted for PyTorch.

Adapted from https://discuss.pytorch.org/t/unet-implementation/426
Implements the original U-Net architecture from Ronneberger et al. (2015).
"""

import torch
import torch.nn.functional as F
from torch import nn


class RonnebergerUNet(nn.Module):
    """Original Ronneberger U-Net architecture for biomedical image segmentation.

    Implementation of U-Net: Convolutional Networks for Biomedical Image
    Segmentation (Ronneberger et al., 2015).
    See https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=3,
        wf=6,
        padding=True,
        batch_norm=False,
        up_mode='upconv',
    ):
        """Initialize the Ronneberger UNet.

        Implementation of U-Net: Convolutional Networks for Biomedical
        Image Segmentation (Ronneberger et al., 2015).
        See https://arxiv.org/abs/1505.04597

        Using the default arguments yields the exact version used in
        the original paper.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        n_classes : int
            Number of output channels.
        depth : int
            Depth of the network (number of encoder levels).
        wf : int
            Width factor; the number of filters in the first layer
            is ``2**wf``.
        padding : bool
            If ``True``, apply padding so that input and output shapes
            match. May introduce artifacts.
        batch_norm : bool
            Whether to use BatchNorm after activation functions.
        up_mode : str
            Upsampling mode: ``'upconv'`` for transposed convolutions
            or ``'upsample'`` for bilinear upsampling.
        """
        super().__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    """Double convolution block for the UNet.

    Two 3x3 convolutions each followed by ReLU and optional batch
    normalization.

    Parameters
    ----------
    in_size : int
        Number of input channels.
    out_size : int
        Number of output channels.
    padding : bool
        Whether to pad convolutions.
    batch_norm : bool
        Whether to apply batch normalization.
    """

    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    """Upsampling block for the UNet decoder path.

    Upsamples the input and concatenates with the skip connection
    from the encoder, then applies a convolution block.

    Parameters
    ----------
    in_size : int
        Number of input channels.
    out_size : int
        Number of output channels.
    up_mode : str
        Upsampling mode: ``'upconv'`` or ``'upsample'``.
    padding : bool
        Whether to pad convolutions.
    batch_norm : bool
        Whether to apply batch normalization.
    """

    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    # def center_crop(self, layer, target_size):
    #     _, _, layer_height, layer_width = layer.size()
    #     diff_y = (layer_height - target_size[0]) // 2
    #     diff_x = (layer_width - target_size[1]) // 2
    #     return layer[
    #            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
    #            ]

    def forward(self, x, bridge):
        up = self.up(x)
        # crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out
