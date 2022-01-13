import torch
from torch import nn

from aydin.nn.layers.conv_with_batch_norm import ConvWithBatchNorm
from aydin.nn.layers.pooling_down import PoolingDown


class UNetModel(nn.Module):
    def __init__(
        self,
        spacetime_ndim,
        nb_unet_levels: int = 4,
        learning_rate=0.01,
        supervised: bool = False,
        pooling_mode: str = 'max',
    ):
        self.learning_rate = learning_rate
        self.supervised = supervised
        self.nb_unet_levels = nb_unet_levels

        self.conv_with_batch_norm = ConvWithBatchNorm(spacetime_ndim)
        self.pooling_down = PoolingDown(spacetime_ndim, pooling_mode)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv = torch.conv2d() if spacetime_ndim == 2 else torch.conv3d()
        self.maskout = None  # TODO: assign correct maskout module

    def forward(self, x):

        # TODO: implement the first skiplayer here

        for layer_index in range(self.nb_unet_levels):
            if layer_index == 0:
                x = self.conv_with_batch_norm(x)

            x = self.conv_with_batch_norm(x)

            x = self.pooling_down(x)

        x = self.conv_with_batch_norm(x)

        for layer_index in range(self.nb_unet_levels):
            x = self.upsampling(x)

            # if self.residual:
            #     x = self.add()
            # else:
            #     x = torch.cat()

            x = self.conv_with_batch_norm(x)

            x = self.conv_with_batch_norm(x)

        x = self.conv(x)

        if not self.supervised:
            x = self.maskout(x)

        return x
