import torch
from torch import nn

from aydin.nn.layers.conv_with_batch_norm import ConvWithBatchNorm
from aydin.nn.layers.pooling_down import PoolingDown


class UNetModel(nn.Module):
    def __init__(
        self,
        spacetime_ndim,
        nb_unet_levels: int = 4,
        nb_filters: int = 8,
        learning_rate=0.01,
        supervised: bool = False,
        pooling_mode: str = 'max',
    ):
        self.nb_unet_levels = nb_unet_levels
        self.nb_filters = nb_filters
        self.learning_rate = learning_rate
        self.supervised = supervised

        self.conv_with_batch_norms_first_half = []
        for layer_index in range(self.nb_unet_levels):
            if (
                layer_index == 0
            ):  # Handle special case input dimensions for the first layer
                self.conv_with_batch_norms_first_half.append(
                    ConvWithBatchNorm(1, self.nb_filters, spacetime_ndim)
                )
            else:
                self.conv_with_batch_norms_first_half.append(
                    ConvWithBatchNorm(
                        self.nb_filters * layer_index,
                        self.nb_filters * (layer_index + 1),
                        spacetime_ndim,
                    )
                )

        self.unet_bottom_conv_out_channels = (
            self.nb_filters,
        )  # TODO: check if it is a bug to use this filter size on the bottom

        self.unet_bottom_conv_with_batch_norm = ConvWithBatchNorm(
            self.conv_with_batch_norms_first_half[-1].out_channels,
            self.unet_bottom_conv_out_channels,
            spacetime_ndim,
        )

        self.conv_with_batch_norms_second_half = []
        for layer_index in range(self.nb_unet_levels):
            if (
                layer_index == 0
            ):  # Handle special case input dimensions for the first layer
                self.conv_with_batch_norms_second_half.append(
                    ConvWithBatchNorm(
                        self.unet_bottom_conv_out_channels,
                        self.nb_filters
                        * max((self.nb_unet_levels - layer_index - 2), 1),
                        spacetime_ndim,
                    )
                )
            else:
                self.conv_with_batch_norms_second_half.append(
                    ConvWithBatchNorm(
                        self.nb_filters
                        * max((self.nb_unet_levels - layer_index - 1 - 2), 1),
                        self.nb_filters
                        * max((self.nb_unet_levels - layer_index - 2), 1),
                        spacetime_ndim,
                    )
                )

        self.pooling_down = PoolingDown(spacetime_ndim, pooling_mode)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv = torch.conv2d if spacetime_ndim == 2 else torch.conv3d
        self.maskout = None  # TODO: assign correct maskout module

    def forward(self, x):

        skip_layer = [x]

        for layer_index in range(self.nb_unet_levels):
            if layer_index == 0:
                x = self.conv_with_batch_norms_first_half[layer_index](x)

            x = self.conv_with_batch_norms_first_half[layer_index](x)

            x = self.pooling_down(x)

            if layer_index != (self.nb_unet_levels - 1):
                skip_layer.append(x)

        x = self.unet_bottom_conv_with_batch_norm(x)

        for layer_index in range(self.nb_unet_levels):
            x = self.upsampling(x)

            if self.residual:
                x = torch.add(x, skip_layer.pop())
            else:
                x = torch.cat([x, skip_layer.pop()])

            # x = self.conv_with_batch_norms_first_half[](x)
            #
            # x = self.conv_with_batch_norms_first_half[](x)

        x = self.conv(x)

        if not self.supervised:
            x = self.maskout(x)

        return x
