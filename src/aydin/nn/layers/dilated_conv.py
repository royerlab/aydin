"""Dilated convolution layer with zero-padding and activation.

Provides a dilated convolution building block used in J-invariant network
(JINet) architectures to increase the receptive field without losing
resolution.
"""

from torch import nn
from torch.nn import ConstantPad2d, ConstantPad3d


class DilatedConv(nn.Module):
    """Dilated convolution with explicit zero-padding and activation.

    Applies zero-padding followed by a dilated convolution and an
    activation function. Supports both 2D and 3D inputs.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    spacetime_ndim : int
        Number of spatial dimensions (2 or 3).
    padding : int
        Amount of zero-padding to apply on each side.
    kernel_size : int
        Size of the convolution kernel.
    dilation : int
        Dilation rate for the convolution.
    activation : str
        Activation function: ``'ReLU'``, ``'swish'``, or ``'lrel'``.

    Raises
    ------
    ValueError
        If ``spacetime_ndim`` is not 2 or 3.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        spacetime_ndim,
        padding,
        kernel_size,
        dilation,
        activation="ReLU",
    ):
        """Initialize the DilatedConv layer.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        spacetime_ndim : int
            Number of spatial dimensions (2 or 3).
        padding : int
            Amount of zero-padding to apply on each side.
        kernel_size : int
            Size of the convolution kernel.
        dilation : int
            Dilation rate for the convolution.
        activation : str
            Activation function: ``'ReLU'``, ``'swish'``, or ``'lrel'``.

        Raises
        ------
        ValueError
            If ``spacetime_ndim`` is not 2 or 3.
        """
        super(DilatedConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spacetime_ndim = spacetime_ndim
        self.activation = activation

        if spacetime_ndim == 2:
            self.conv_class = nn.Conv2d
            self.zero_padding = ConstantPad2d(padding, value=0)
        elif spacetime_ndim == 3:
            self.conv_class = nn.Conv3d
            self.zero_padding = ConstantPad3d(padding, value=0)
        else:
            raise ValueError("spacetime_ndim parameter can only be 2 or 3...")

        self.conv = self.conv_class(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding='valid',
        )

        self.activation_function = {
            "ReLU": nn.ReLU(),
            "swish": nn.SiLU(),
            "lrel": nn.LeakyReLU(0.1),
        }[self.activation]

    def forward(self, x):
        """Apply zero-padding, dilated convolution, and activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after padding, convolution, and activation.
        """
        x = self.zero_padding(x)

        x = self.conv(x)

        x = self.activation_function(x)

        return x
