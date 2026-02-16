"""Custom convolution layers with normalization and activation.

Provides a configurable convolution block and a helper to build
double-convolution sequences, supporting 2D and 3D spatial dimensions.
"""

import torch.nn.functional as F
from torch import nn


class CustomConv(nn.Module):
    """Convolution layer with optional normalization and activation.

    Wraps a 2D or 3D convolution followed by optional instance/batch
    normalization and a configurable activation function.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    spacetime_ndim : int
        Number of spatial dimensions (2 or 3).
    kernel_size : int
        Size of the convolution kernel.
    normalization : str or None
        Normalization type: ``'instance'``, ``'batch'``, or ``None``.
    activation : str
        Activation function: ``'ReLU'``, ``'swish'``, or ``'lrel'``.
    padding_mode : str
        Padding mode: ``'zeros'`` for zero padding (default) or
        ``'reflect'`` for reflection padding (reduces boundary artifacts).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        spacetime_ndim,
        kernel_size=3,
        normalization=None,  # "batch",
        activation="ReLU",
        padding_mode='zeros',
    ):
        """Initialize the CustomConv layer.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        spacetime_ndim : int
            Number of spatial dimensions (2 or 3).
        kernel_size : int
            Size of the convolution kernel.
        normalization : str or None
            Normalization type: ``'instance'``, ``'batch'``, or ``None``.
        activation : str
            Activation function: ``'ReLU'``, ``'swish'``, or ``'lrel'``.
        padding_mode : str
            Padding mode: ``'zeros'`` or ``'reflect'``.
        """
        super(CustomConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spacetime_ndim = spacetime_ndim
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.activation = activation
        self.padding_mode = padding_mode

        pad_amount = kernel_size // 2

        if padding_mode == 'reflect' and spacetime_ndim == 2:
            self.reflect_pad = nn.ReflectionPad2d(pad_amount)
            self.conv = nn.Conv2d(
                in_channels, out_channels, (kernel_size,) * 2, padding=0
            )
        elif padding_mode == 'reflect' and spacetime_ndim == 3:
            # Store pad amounts for F.pad in forward pass
            self._reflect_pad_amounts = (pad_amount,) * 6  # D, H, W each side
            self.reflect_pad = None
            self.conv = nn.Conv3d(
                in_channels, out_channels, (kernel_size,) * 3, padding=0
            )
        elif spacetime_ndim == 2:
            self.reflect_pad = None
            self.conv = nn.Conv2d(
                in_channels, out_channels, (kernel_size,) * 2, padding='same'
            )
        else:
            self.reflect_pad = None
            self.conv = nn.Conv3d(
                in_channels, out_channels, (kernel_size,) * 3, padding='same'
            )

        if spacetime_ndim == 2:
            self.instance_normalization = nn.InstanceNorm2d(out_channels)
            self.batch_normalization = nn.BatchNorm2d(out_channels, affine=False)
        else:
            self.instance_normalization = nn.InstanceNorm3d(out_channels)
            self.batch_normalization = nn.BatchNorm3d(out_channels, affine=False)

        self.activation_function = {
            "ReLU": nn.ReLU(),
            "swish": nn.SiLU(),
            "lrel": nn.LeakyReLU(0.1),
        }[self.activation]

    def forward(self, x):
        """Apply convolution, normalization, and activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after convolution, normalization, and activation.
        """
        if self.padding_mode == 'reflect':
            if self.reflect_pad is not None:
                x = self.reflect_pad(x)
            else:
                x = F.pad(x, self._reflect_pad_amounts, mode='reflect')

        x = self.conv(x)

        if self.normalization == 'instance':
            x = self.instance_normalization(x)
        elif self.normalization == 'batch':
            x = self.batch_normalization(x)

        x = self.activation_function(x)

        return x


def double_conv_block(
    nb_filters_in,
    nb_filters_inner,
    nb_filters_out,
    spacetime_ndim,
    normalizations=(None, None),
    padding_mode='zeros',
):
    """Create a sequential block of two CustomConv layers.

    Parameters
    ----------
    nb_filters_in : int
        Number of input filters for the first convolution.
    nb_filters_inner : int
        Number of output filters for the first convolution and input
        filters for the second.
    nb_filters_out : int
        Number of output filters for the second convolution.
    spacetime_ndim : int
        Number of spatial dimensions (2 or 3).
    normalizations : tuple of (str or None)
        Normalization types for the first and second convolutions.
    padding_mode : str
        Padding mode: ``'zeros'`` or ``'reflect'``.

    Returns
    -------
    torch.nn.Sequential
        Sequential module containing two CustomConv layers.
    """
    return nn.Sequential(
        CustomConv(
            nb_filters_in,
            nb_filters_inner,
            spacetime_ndim,
            normalization=normalizations[0],
            padding_mode=padding_mode,
        ),
        CustomConv(
            nb_filters_inner,
            nb_filters_out,
            spacetime_ndim,
            normalization=normalizations[1],
            padding_mode=padding_mode,
        ),
    )
