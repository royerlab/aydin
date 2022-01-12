import torch
from torch import nn


class UNetModel(nn.Module):
    def __init__(
        self,
        spacetime_ndim,
        nb_unet_levels: int = 4,
        learning_rate=0.01,
        supervised: bool = False,
        shiftconv: bool = True,
        pooling_mode: str = 'max',
    ):
        self.learning_rate = learning_rate
        self.supervised = supervised
        self.shiftconv = shiftconv
        self.nb_unet_levels = nb_unet_levels

        self.custom_rot90 = CustomRot90(
            spacetime_ndim
        )  # TODO: assign correct custom module here for both 2d and 3d
        self.conv_with_batch_norm = ConvWithBatchNorm(spacetime_ndim)
        self.pooling_down = (
            PoolingDown2D(shiftconv, pooling_mode)
            if spacetime_ndim == 2
            else PoolingDown3D(shiftconv, pooling_mode)
        )
        self.upsampling = (
            UpSampling2D(shiftconv, upsampling_mode)
            if spacetime_ndim == 2
            else UpSampling3D(shiftconv, upsampling_mode)
        )
        self.conv = torch.conv2d() if spacetime_ndim == 2 else torch.conv3d()
        self.maskout = None  # TODO: assign correct maskout module

    def forward(self, x):

        if self.shiftconv:
            x = self.custom_rot90(x)

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

        if self.shiftconv:
            # TODO: handle special case
            pass

        x = self.conv(x)

        if not self.shiftconv and not self.supervised:
            x = self.maskout(x)

        return x
