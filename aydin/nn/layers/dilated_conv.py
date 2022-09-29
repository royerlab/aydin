from torch import nn
from torch.nn import ConstantPad2d, ConstantPad3d


class DilatedConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spacetime_ndim,
        padding,
        kernel_size,
        dilation,
        activation="ReLU",
    ):
        super(DilatedConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spacetime_ndim = spacetime_ndim
        self.activation = activation

        if spacetime_ndim == 2:
            self.conv_class = nn.Conv2d
            self.zero_padding = ConstantPad2d(padding, value=0)
        elif spacetime_ndim == 3:
            self.conv_class = nn.Conv3d
            self.zero_padding = ConstantPad3d(padding, value=0)
        else:
            raise ValueError("spacetime_ndim parameter can only be 2 or 3...")

        self.conv = self.conv_class(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding='valid',
        )

        self.activation_function = {
            "ReLU": nn.ReLU(),
            "swish": nn.SiLU(),
            "lrel": nn.LeakyReLU(0.1),
        }[self.activation]

    def forward(self, x):
        x = self.zero_padding(x)

        x = self.conv(x)

        x = self.activation_function(x)

        return x
