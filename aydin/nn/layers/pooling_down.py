from torch import nn


class PoolingDown(nn.Module):
    def __init__(self, spacetime_ndim, shiftconv, pooling_mode):
        self.spacetime_ndim = spacetime_ndim
        self.shiftconv = shiftconv
        self.pooling_mode = pooling_mode

        self.zero_padding = None  # TODO: assign right thing for either 2d or 3d
        self.cropping = None  # TODO: assign right thing for either 2d or 3d
        self.average_pooling = None  # TODO: assign right thing for either 2d or 3d
        self.max_pooling = None  # TODO: assign right thing for either 2d or 3d

    def forward(self, x):
        if self.shiftconv:
            x = self.zero_padding(x)
            x = self.cropping(x)

        if self.pooling_mode == 'ave':
            x = self.average_pooling(x)
        elif self.pooling_mode == 'max':
            x = self.max_pooling(x)
        else:
            raise ValueError('pooling mode only accepts "max" or "ave".')

        return x
