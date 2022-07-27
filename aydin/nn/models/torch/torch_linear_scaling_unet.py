from torch import nn

from aydin.nn.layers.pooling_down import PoolingDown


class LinearScalingUNetModel(nn.Module):
    def __init__(
            self,
            spacetime_ndim,
            nb_unet_levels: int = 4,
            nb_filters: int = 8,
            learning_rate=0.01,
            supervised: bool = False,
            pooling_mode: str = 'max',
    ):
        super(LinearScalingUNetModel, self).__init__()

        self.spacetime_ndim = spacetime_ndim
        self.nb_unet_levels = nb_unet_levels
        self.nb_filters = nb_filters
        self.learning_rate = learning_rate
        self.supervised = supervised
        self.pooling_down = PoolingDown(spacetime_ndim, pooling_mode)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')

        self.unet_bottom_conv_out_channels = self.nb_filters * (self.nb_unet_levels - 1)

    def forward(self, x):
        return x
