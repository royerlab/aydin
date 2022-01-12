from torch import nn


class ConvWithBatchNorm(nn.Module):
    def __init__(
        self, spacetime_ndim, shiftconv, normalization=None, activation="ReLU"
    ):
        self.spacetime_ndim = spacetime_ndim
        self.shiftconv = shiftconv
        self.normalization = normalization
        self.activation = activation

        self.ZeroPadding = None  # TODO: assign right thing for 2d or 3d
        self.conv = None  # TODO: assign right thing for 2d or 3d
        self.Cropping = None  # TODO: assign right thing for 2d or 3d
        self.relu = nn.ReLU()
        self.swish = nn.SiLU()
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        if self.shiftconv:
            x = self.ZeroPadding(x)
            x = self.conv(x)
            x = self.Cropping(x)
        else:
            x = self.conv(x)

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
