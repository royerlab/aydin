"""DnCNN (Denoising Convolutional Neural Network) implementation.

Implements the DnCNN architecture from Zhang et al. (2017),
a feed-forward denoising network with batch normalization.
"""

import torch.nn as nn


class DnCNN(nn.Module):
    """DnCNN denoising network.

    Feed-forward CNN with batch normalization for image denoising,
    based on the architecture from Zhang et al. (2017).

    Parameters
    ----------
    n_channel_in : int
        Number of input channels.
    n_channel_out : int
        Number of output channels.
    num_of_layers : int
        Total number of convolution layers.
    kernel_size : int
        Convolution kernel size.
    padding : int
        Padding size for convolutions.
    features : int
        Number of intermediate feature channels.
    """

    def __init__(
        self,
        n_channel_in=1,
        n_channel_out=1,
        num_of_layers=17,
        kernel_size=3,
        padding=1,
        features=64,
    ):
        super(DnCNN, self).__init__()

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=n_channel_in,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                in_channels=features,
                out_channels=n_channel_out,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
