"""Simple feed-forward CNN with residual connections.

Provides a straightforward convolutional architecture where outputs
from all layers are summed with residual connections.
"""

import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Feed-forward CNN with residual connections.

    A simple convolutional network where outputs from all intermediate
    layers are summed together before a final 1x1 convolution.

    Parameters
    ----------
    n_input_channel : int
        Number of input channels.
    n_output_channel : int
        Number of output channels.
    depth : int
        Number of convolution layers.
    nic : int
        Number of intermediate channels.
    kernel_size : int
        Convolution kernel size.
    """

    def __init__(
        self, n_input_channel=1, n_output_channel=1, depth=8, nic=8, kernel_size=3
    ):
        super().__init__()

        self.convs = []

        for i in range(0, depth - 1):
            in_channels = n_input_channel if i == 0 else nic
            conv = nn.Conv2d(
                in_channels,
                nic,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            )
            self.convs.append(conv)

        self.convs = nn.ModuleList(self.convs)
        self.final_conv = nn.Conv2d(nic, n_output_channel, kernel_size=1, padding=0)

    def forward(self, x0):

        x = x0

        xn = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, negative_slope=0.01)
            xn.append(x)

        y = xn[0]
        s = 1
        for x in xn[1:]:
            y = y + s * x
            # s*=0.5

        y = self.final_conv(y)

        return y
