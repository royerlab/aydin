from torch import nn


class PoolingDown(nn.Module):
    def __init__(self, spacetime_ndim, pooling_mode):
        super(PoolingDown, self).__init__()

        self.spacetime_ndim = spacetime_ndim
        self.pooling_mode = pooling_mode

        if spacetime_ndim == 2:
            self.average_pooling = nn.AvgPool2d((2, 2))
            self.max_pooling = nn.MaxPool2d((2, 2))
        else:
            self.average_pooling = nn.AvgPool3d((2, 2, 2))
            self.max_pooling = nn.MaxPool3d((2, 2, 2))

    def forward(self, x):

        if self.pooling_mode == 'ave':
            x = self.average_pooling(x)
        elif self.pooling_mode == 'max':
            x = self.max_pooling(x)
        else:
            raise ValueError('pooling mode only accepts "max" or "ave".')

        return x
