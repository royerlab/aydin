import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
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
