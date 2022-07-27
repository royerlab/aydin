import torch
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

        if spacetime_ndim == 2:
            self.final_conv = nn.Conv2d(self.nb_filters, 1, 1)
        else:
            self.final_conv = nn.Conv3d(self.nb_filters, 1, 1)

    def forward(self, x, input_mask):
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

        # Masking for self-supervised training
        if not self.supervised:
            if input_mask is not None:
                x *= input_mask
            else:
                raise ValueError(
                    "input_msk cannot be None for self-supervised training"
                )

        return x
