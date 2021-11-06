import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=False,
        norm=None,
        residual=True,
        activation='leakyrelu',
        in_place_activation=True,
        transpose=False,
        reflectpad=True,
    ):

        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.residual = residual
        self.activation = activation
        self.transpose = transpose
        self.reflectpad = reflectpad

        if self.dropout:
            self.dropout1 = nn.Dropout2d(p=0.05)
            self.dropout2 = nn.Dropout2d(p=0.05)

        self.norm1 = None
        self.norm2 = None
        if norm is not None:
            if norm == 'batch':
                self.norm1 = nn.BatchNorm2d(out_channels)
                self.norm2 = nn.BatchNorm2d(out_channels)
            elif norm == 'instance':
                self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
                self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

        if self.transpose:
            self.conv1 = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=0 if self.reflectpad else 1,
            )
            self.conv2 = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=0 if self.reflectpad else 1,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=0 if self.reflectpad else 1,
            )
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=0 if self.reflectpad else 1,
            )

        if self.activation == 'relu':
            self.actfun1 = nn.ReLU(inplace=in_place_activation)
            self.actfun2 = nn.ReLU(inplace=in_place_activation)
        elif self.activation == 'leakyrelu':
            self.actfun1 = nn.LeakyReLU(inplace=in_place_activation)
            self.actfun2 = nn.LeakyReLU(inplace=in_place_activation)
        elif self.activation == 'elu':
            self.actfun1 = nn.ELU(inplace=in_place_activation)
            self.actfun2 = nn.ELU(inplace=in_place_activation)
        elif self.activation == 'selu':
            self.actfun1 = nn.SELU(inplace=in_place_activation)
            self.actfun2 = nn.SELU(inplace=in_place_activation)

        if self.reflectpad:
            self.rpad1 = nn.ReflectionPad2d(1)
            self.rpad2 = nn.ReflectionPad2d(1)

    def forward(self, x):

        ox = x

        if self.reflectpad:
            x = self.rpad1(x)

        x = self.conv1(x)

        if self.dropout:
            x = self.dropout1(x)

        x = self.actfun1(x)

        if self.norm1:
            x = self.norm1(x)

        if self.reflectpad:
            x = self.rpad2(x)

        x = self.conv2(x)

        if self.dropout:
            x = self.dropout2(x)

        if self.residual:
            x[:, 0 : min(ox.shape[1], x.shape[1]), :, :] += ox[
                :, 0 : min(ox.shape[1], x.shape[1]), :, :
            ]

        x = self.actfun2(x)

        if self.norm2:
            x = self.norm2(x)

        # print("shapes: x:%s ox:%s " % (x.shape,ox.shape))

        return x
