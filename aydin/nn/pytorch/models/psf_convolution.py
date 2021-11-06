import torch
from torch import nn


class PSFConvolutionLayer(nn.Module):
    def __init__(self, kernel_psf, num_channels_in=1, num_channels_out=1):
        kernel_size = kernel_psf.shape[0]
        super().__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(
                num_channels_in,
                num_channels_out,
                kernel_size,
                stride=1,
                padding=0,
                bias=False,
                groups=num_channels_in,
            ),
        )

        self.weights_init(kernel_psf)

    def weights_init(self, kernel_psf):
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(kernel_psf))

    def forward(self, x):
        return self.seq(x)
