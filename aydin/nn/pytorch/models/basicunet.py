import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = F.leaky_relu(
            self.norm(self.conv2(self.conv1(x))), negative_slope=0.001, inplace=True
        )
        return x


class BasicUNet(nn.Module):
    def __init__(self, n_input_channel=1, n_output_channel=1, nic=8, residual=True):
        super().__init__()

        self.residual = residual

        self.down1 = nn.MaxPool2d(kernel_size=2)
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.down3 = nn.MaxPool2d(kernel_size=2)
        self.down4 = nn.MaxPool2d(kernel_size=2)

        self.up1 = lambda x: nn.functional.interpolate(
            x, mode='nearest', scale_factor=2
        )
        self.up2 = lambda x: nn.functional.interpolate(
            x, mode='nearest', scale_factor=2
        )
        self.up3 = lambda x: nn.functional.interpolate(
            x, mode='nearest', scale_factor=2
        )
        self.up4 = lambda x: nn.functional.interpolate(
            x, mode='nearest', scale_factor=2
        )

        self.conv1 = BasicConvBlock(n_input_channel, nic)

        self.conv2 = BasicConvBlock(nic, nic * 2)
        self.conv3 = BasicConvBlock(nic * 2, nic * 4)
        self.conv4 = BasicConvBlock(nic * 4, nic * 8)
        self.conv5 = BasicConvBlock(nic * 8, nic * 16)

        self.conv6 = BasicConvBlock(nic * 3 * 8, nic * 8)
        self.conv7 = BasicConvBlock(nic * 3 * 4, nic * 4)
        self.conv8 = BasicConvBlock(nic * 3 * 2, nic * 2)
        self.conv9 = BasicConvBlock(nic * 3, nic)

        self.conv10 = BasicConvBlock(nic, n_output_channel)

        if self.residual:
            self.conv11 = BasicConvBlock(n_input_channel, n_output_channel)

    def forward(self, x):
        c0 = x
        c1 = self.conv1(x)
        x = self.down1(c1)
        c2 = self.conv2(x)
        x = self.down2(c2)
        c3 = self.conv3(x)
        x = self.down3(c3)
        c4 = self.conv4(x)
        x = self.down4(c4)
        x = self.conv5(x)
        x = self.up1(x)
        x = torch.cat([x, c4], 1)
        x = self.conv6(x)
        x = self.up2(x)
        x = torch.cat([x, c3], 1)
        x = self.conv7(x)
        x = self.up3(x)
        x = torch.cat([x, c2], 1)
        x = self.conv8(x)
        x = self.up4(x)
        x = torch.cat([x, c1], 1)
        x = self.conv9(x)
        x = self.conv10(x)
        if self.residual:
            x = torch.add(x, self.conv11(c0))

        return x
