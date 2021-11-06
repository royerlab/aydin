import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(
        self,
        n_channel_in=1,
        n_channel_out=1,
        num_of_layers=17,
        kernel_size=3,
        padding=1,
        features=64,
    ):
        super(DnCNN, self).__init__()

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=n_channel_in,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                in_channels=features,
                out_channels=n_channel_out,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
