from torch import nn


class ConvWithBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spacetime_ndim,
        kernel_size=3,
        normalization="batch",
        activation="ReLU",
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spacetime_ndim = spacetime_ndim
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.activation = activation

        self.conv = (
            nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size))
            if spacetime_ndim == 2
            else nn.Conv3d(
                in_channels, out_channels, (kernel_size, kernel_size, kernel_size)
            )
        )
        self.relu = nn.ReLU()
        self.swish = nn.SiLU()
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x, kernel_size=self.kernel_size)

        if self.normalization == 'instance':
            x = self.instance_normalization(x)
        elif self.normalization == 'batch':
            x = self.batch_normalization(x)

        if self.activation == 'ReLU':
            x = self.relu(x)
        elif self.activation == 'swish':
            x = self.swish(x)
        elif self.activation == 'lrel':
            x = self.leaky_relu(x)

        return x
