"""Ronneberger UNet architecture with 2D and 3D support.

Implements the original U-Net from Ronneberger et al. (2015) for
biomedical image segmentation, adapted for image denoising with
support for both 2D and 3D spatial inputs.

Reference: https://arxiv.org/abs/1505.04597
"""

import torch
import torch.nn.functional as F
from torch import nn


class RonnebergerUNetModel(nn.Module):
    """Original Ronneberger U-Net architecture for image denoising.

    Encoder-decoder network with skip connections following the
    original U-Net design. Supports configurable depth, width factor,
    batch normalization, and upsampling mode.
    <notgui>
    """

    def __init__(
        self,
        spacetime_ndim,
        depth=3,
        wf=6,
        padding=True,
        batch_norm=False,
        up_mode='upconv',
    ):
        """Initialize the Ronneberger UNet model.

        Parameters
        ----------
        spacetime_ndim : int
            Number of spatial dimensions (2 or 3).
        depth : int
            Depth of the network (number of encoder levels).
        wf : int
            Width factor; the number of filters in the first level
            is ``2**wf``.
        padding : bool
            If ``True``, apply padding so input and output shapes match.
        batch_norm : bool
            Whether to use BatchNorm after activation functions.
        up_mode : str
            Upsampling mode: ``'upconv'`` for transposed convolutions
            or ``'upsample'`` for bilinear/trilinear upsampling.

        Raises
        ------
        ValueError
            If ``spacetime_ndim`` is not 2 or 3 or ``up_mode`` is invalid.
        """
        super().__init__()

        if spacetime_ndim not in (2, 3):
            raise ValueError("spacetime_ndim must be 2 or 3")
        if up_mode not in ('upconv', 'upsample'):
            raise ValueError("up_mode must be 'upconv' or 'upsample'")

        self.spacetime_ndim = spacetime_ndim
        self.nb_unet_levels = depth
        self.padding = padding
        self.depth = depth

        prev_channels = 1  # single input channel for denoising
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                _RonnebergerConvBlock(
                    prev_channels,
                    2 ** (wf + i),
                    spacetime_ndim,
                    padding,
                    batch_norm,
                )
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                _RonnebergerUpBlock(
                    prev_channels,
                    2 ** (wf + i),
                    spacetime_ndim,
                    up_mode,
                    padding,
                    batch_norm,
                )
            )
            prev_channels = 2 ** (wf + i)

        if spacetime_ndim == 2:
            self.last = nn.Conv2d(prev_channels, 1, kernel_size=1)
        else:
            self.last = nn.Conv3d(prev_channels, 1, kernel_size=1)

    def forward(self, x):
        """Run the forward pass through the Ronneberger UNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(B, 1, ...spatial_dims...)``.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as the input.
        """
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                if self.spacetime_ndim == 2:
                    x = F.max_pool2d(x, 2)
                else:
                    x = F.max_pool3d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class _RonnebergerConvBlock(nn.Module):
    """Double convolution block for the Ronneberger UNet.

    Two convolutions each followed by ReLU and optional batch
    normalization.
    """

    def __init__(self, in_size, out_size, spacetime_ndim, padding, batch_norm):
        """Initialize the double convolution block.

        Parameters
        ----------
        in_size : int
            Number of input channels.
        out_size : int
            Number of output channels.
        spacetime_ndim : int
            Spatial dimensionality (2 or 3).
        padding : bool
            Whether to apply same-padding to convolutions.
        batch_norm : bool
            Whether to apply batch normalization after each convolution.
        """
        super().__init__()

        if spacetime_ndim == 2:
            conv_class = nn.Conv2d
            bn_class = nn.BatchNorm2d
        else:
            conv_class = nn.Conv3d
            bn_class = nn.BatchNorm3d

        block = []
        block.append(conv_class(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(bn_class(out_size))

        block.append(
            conv_class(out_size, out_size, kernel_size=3, padding=int(padding))
        )
        block.append(nn.ReLU())
        if batch_norm:
            block.append(bn_class(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        """Apply the double convolution block to the input tensor."""
        return self.block(x)


class _RonnebergerUpBlock(nn.Module):
    """Upsampling block for the Ronneberger UNet decoder path.

    Upsamples the input and concatenates with the skip connection
    from the encoder, then applies a convolution block.
    """

    def __init__(self, in_size, out_size, spacetime_ndim, up_mode, padding, batch_norm):
        """Initialize the upsampling block.

        Parameters
        ----------
        in_size : int
            Number of input channels.
        out_size : int
            Number of output channels.
        spacetime_ndim : int
            Spatial dimensionality (2 or 3).
        up_mode : str
            Upsampling mode: 'upconv' for transposed convolution,
            'upsample' for interpolation followed by 1x1 convolution.
        padding : bool
            Whether to apply same-padding in the convolution block.
        batch_norm : bool
            Whether to apply batch normalization.
        """
        super().__init__()

        if up_mode == 'upconv':
            if spacetime_ndim == 2:
                self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
            else:
                self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            upsample_mode = 'bilinear' if spacetime_ndim == 2 else 'trilinear'
            if spacetime_ndim == 2:
                self.up = nn.Sequential(
                    nn.Upsample(mode=upsample_mode, scale_factor=2),
                    nn.Conv2d(in_size, out_size, kernel_size=1),
                )
            else:
                self.up = nn.Sequential(
                    nn.Upsample(mode=upsample_mode, scale_factor=2),
                    nn.Conv3d(in_size, out_size, kernel_size=1),
                )

        self.conv_block = _RonnebergerConvBlock(
            in_size, out_size, spacetime_ndim, padding, batch_norm
        )

    def forward(self, x, bridge):
        """Upsample input and concatenate with encoder skip connection."""
        up = self.up(x)
        # Pad if spatial dimensions differ (happens with odd input sizes)
        if up.shape[2:] != bridge.shape[2:]:
            padding = []
            for i in range(len(up.shape) - 1, 1, -1):
                diff = bridge.shape[i] - up.shape[i]
                padding.extend([diff // 2, diff - diff // 2])
            up = torch.nn.functional.pad(up, padding)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out
