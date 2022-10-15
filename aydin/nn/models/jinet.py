import torch
from torch import nn
from aydin.nn.layers.dilated_conv import DilatedConv


class JINetModel(nn.Module):
    def __init__(
        self,
        spacetime_ndim,
        nb_in_channels: int = 1,
        nb_out_channels: int = 1,
        kernel_sizes=None,
        num_features=None,
        nb_dense_layers: int = 3,
        nb_channels: int = None,
        final_relu: bool = False,
        degressive_residuals: bool = False,  # TODO: check what happens when this is True
    ):
        super(JINetModel, self).__init__()

        self.spacetime_ndim = spacetime_ndim
        self.nb_in_channels = nb_in_channels
        self.nb_out_channels = nb_out_channels
        self._kernel_sizes = kernel_sizes
        self._num_features = num_features
        self.nb_dense_layers = nb_dense_layers
        self.nb_channels = nb_channels
        self.final_relu = final_relu
        self.degressive_residuals = degressive_residuals

        if len(self.kernel_sizes) != len(self.num_features):
            raise ValueError("Number of kernel sizes and features does not match.")

        self.dilated_conv_functions = nn.ModuleList()
        current_receptive_field_radius = 0
        for scale_index in range(len(self.kernel_sizes)):
            # Get kernel size and number of features:
            kernel_size = self.kernel_sizes[scale_index]

            # radius and dilation:
            radius = (kernel_size - 1) // 2
            dilation = 1 + current_receptive_field_radius

            self.dilated_conv_functions.append(
                DilatedConv(
                    self.nb_in_channels
                    if scale_index == 0
                    else self.num_features[scale_index - 1],
                    self.num_features[scale_index],
                    self.spacetime_ndim,
                    padding=dilation * radius,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation="lrel",
                )
            )

            # update receptive field radius
            current_receptive_field_radius += dilation * radius

        if spacetime_ndim == 2:
            self.conv = nn.Conv2d
        elif spacetime_ndim == 3:
            self.conv = nn.Conv3d
        else:
            raise ValueError("spacetime_ndim can not be anything other than 2 or 3...")

        if self.nb_channels is None:
            self.nb_channels = sum(self.num_features)  # * 2

        nb_out = self.nb_channels
        self.kernel_one_conv_functions = nn.ModuleList()
        for index in range(self.nb_dense_layers):
            nb_in = nb_out
            nb_out = (
                self.nb_out_channels
                if index == (self.nb_dense_layers - 1)
                else self.nb_channels
            )

            self.kernel_one_conv_functions.append(
                self.conv(
                    in_channels=nb_in,
                    out_channels=nb_out,
                    kernel_size=(1,) * spacetime_ndim,
                )
            )

        self.final_kernel_one_conv = self.conv(
            in_channels=self.nb_channels,
            out_channels=1,
            kernel_size=(1,) * spacetime_ndim,
        )

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    @property
    def kernel_sizes(self):
        if self._kernel_sizes is None:
            if self.spacetime_ndim == 2:
                self._kernel_sizes = [7, 5, 3, 3, 3, 3, 3, 3]
            elif self.spacetime_ndim == 3:
                self._kernel_sizes = [7, 5, 3, 3]

        return self._kernel_sizes

    @property
    def num_features(self):
        if self._num_features is None:
            if self.spacetime_ndim == 2:
                self._num_features = [64, 32, 16, 8, 4, 2, 1, 1]
            elif self.spacetime_ndim == 3:
                self._num_features = [10, 8, 4, 2]

        return self._num_features

    def forward(self, x):
        dilated_conv_list = []

        # Calculate dilated convolutions
        for index in range(len(self.kernel_sizes)):
            x = self.dilated_conv_functions[index](x)
            dilated_conv_list.append(x)
            # print(x.shape)

        # Concat the results
        x = torch.cat(dilated_conv_list, dim=1)
        # print(f"after cat: {x.shape}")

        # First kernel size one conv
        x = self.kernel_one_conv_functions[0](x)
        # print(f"after first kernel one conv: {x.shape}")
        x = self.lrelu(x)
        y = x
        f = 1

        # Rest of the kernel size one convolutions
        for index in range(1, self.nb_dense_layers):
            x = self.kernel_one_conv_functions[index](x)
            x = self.lrelu(x)
            y = y + f * x

            if self.degressive_residuals:
                f = f * 0.5

        # Final kernel size one convolution
        y = self.final_kernel_one_conv(y)

        # Final ReLU
        if self.final_relu:
            y = self.relu(y)

        return y
