import torch
from torch import nn

from aydin.nn.layers.custom_conv import double_conv_block
from aydin.nn.layers.pooling_down import PoolingDown


class UNetModel(nn.Module):
    def __init__(
        self,
        spacetime_ndim,
        nb_unet_levels: int = 4,
        nb_filters: int = 8,
        learning_rate=0.01,
        pooling_mode: str = 'max',
    ):
        super(UNetModel, self).__init__()

        self.spacetime_ndim = spacetime_ndim
        self.nb_unet_levels = nb_unet_levels
        self.nb_filters = nb_filters
        self.learning_rate = learning_rate
        self.pooling_down = PoolingDown(spacetime_ndim, pooling_mode)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')

        self.double_conv_blocks_encoder = self._encoder_convolutions()

        self.unet_bottom_conv_out_channels = self.nb_filters * (
            2 ** (self.nb_unet_levels - 1)
        )
        self.unet_bottom_conv_block = double_conv_block(
            self.unet_bottom_conv_out_channels,
            self.unet_bottom_conv_out_channels * 2,
            self.unet_bottom_conv_out_channels,
            spacetime_ndim,
        )

        self.double_conv_blocks_decoder = self._decoder_convolutions()

        if spacetime_ndim == 2:
            self.final_conv = nn.Conv2d(self.nb_filters, 1, 1)
        else:
            self.final_conv = nn.Conv3d(self.nb_filters, 1, 1)

    def forward(self, x):
        """
        UNet forward method.

        Parameters
        ----------
        x
        input_mask : numpy.ArrayLike
            A mask per image must be passed with self-supervised training.

        Returns
        -------

        """
        skip_layer = []

        # Encoder
        for layer_index in range(self.nb_unet_levels):
            x = self.double_conv_blocks_encoder[layer_index](x)
            skip_layer.append(x)
            x = self.pooling_down(x)

        # Bottom
        x = self.unet_bottom_conv_block(x)

        # Decoder
        for layer_index in range(self.nb_unet_levels):
            x = self.upsampling(x)
            x = torch.cat([x, skip_layer.pop()], dim=1)
            x = self.double_conv_blocks_decoder[layer_index](x)

        # Final convolution
        x = self.final_conv(x)

        return x

    def _encoder_convolutions(self):
        convolution = nn.ModuleList()
        for layer_index in range(self.nb_unet_levels):
            if layer_index == 0:
                nb_filters_in = 1
                nb_filters_inner = self.nb_filters
                nb_filters_out = self.nb_filters
            else:
                nb_filters_in = self.nb_filters * (2 ** (layer_index - 1))
                nb_filters_inner = self.nb_filters * (2**layer_index)
                nb_filters_out = self.nb_filters * (2**layer_index)

            convolution.append(
                double_conv_block(
                    nb_filters_in,
                    nb_filters_inner,
                    nb_filters_out,
                    self.spacetime_ndim,
                )
            )

        return convolution

    def _decoder_convolutions(self):
        convolutions = nn.ModuleList()
        for layer_index in range(self.nb_unet_levels):
            if layer_index == self.nb_unet_levels - 1:
                nb_filters_in = self.nb_filters * 2
                nb_filters_inner = nb_filters_out = self.nb_filters
            else:
                nb_filters_in = self.nb_filters * (
                    2 ** (self.nb_unet_levels - layer_index)
                )
                nb_filters_inner = nb_filters_in // 2
                nb_filters_out = nb_filters_in // 4

            convolutions.append(
                double_conv_block(
                    nb_filters_in,
                    nb_filters_inner,
                    nb_filters_out,
                    spacetime_ndim=self.spacetime_ndim,
                    normalizations=(None, "batch"),
                )
            )

        return convolutions
