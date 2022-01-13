from torch import nn


class PoolingDown(nn.Module):
    def __init__(self, spacetime_ndim, pooling_mode):
        self.spacetime_ndim = spacetime_ndim
        self.pooling_mode = pooling_mode

        self.average_pooling = None  # TODO: assign right thing for either 2d or 3d
        self.max_pooling = None  # TODO: assign right thing for either 2d or 3d

    def forward(self, x):

        if self.pooling_mode == 'ave':
            x = self.average_pooling(x)
        elif self.pooling_mode == 'max':
            x = self.max_pooling(x)
        else:
            raise ValueError('pooling mode only accepts "max" or "ave".')

        return x
