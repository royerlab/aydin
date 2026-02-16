"""JINet (J-Invariant Network) model architecture in PyTorch.

Implements a blind-spot CNN using dilated convolutions to achieve
J-invariance for self-supervised (Noise2Self) image denoising.
"""

import numpy
import torch
import torch.nn.functional as F
from torch import nn

from aydin.nn.layers.dilated_conv import DilatedConv


class JINetModel(nn.Module):
    """J-Invariant Network using dilated convolutions for blind-spot denoising.

    Uses a series of dilated convolutions with increasing receptive fields
    followed by 1x1 convolutions (dense layers) with residual connections.
    The architecture inherently excludes the center pixel from the
    receptive field, enabling self-supervised denoising.
    <notgui>
    """

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
        degressive_residuals: bool = True,
        kernel_continuity_regularisation: bool = True,
    ):
        """Initialize the JINet model.

        Parameters
        ----------
        spacetime_ndim : int
            Number of spatial dimensions (2 or 3).
        nb_in_channels : int
            Number of input channels.
        nb_out_channels : int
            Number of output channels.
        kernel_sizes : list of int or None
            Kernel sizes for each dilated convolution scale. If ``None``,
            uses default sizes based on ``spacetime_ndim``.
        num_features : list of int or None
            Number of output features per dilated convolution scale.
            Must have the same length as ``kernel_sizes``. If ``None``,
            uses defaults based on ``spacetime_ndim``.
        nb_dense_layers : int
            Number of 1x1 convolution (dense) layers after feature
            extraction.
        nb_channels : int or None
            Number of channels in the dense layers. If ``None``,
            defaults to the sum of all ``num_features``.
        final_relu : bool
            Whether to apply ReLU activation to the output.
        degressive_residuals : bool
            Whether to apply exponentially decaying weights to residual
            connections in the dense layers.
        kernel_continuity_regularisation : bool
            Whether to apply kernel smoothing regularization when
            ``post_optimisation()`` is called.

        Raises
        ------
        ValueError
            If ``kernel_sizes`` and ``num_features`` have different
            lengths, or if ``spacetime_ndim`` is not 2 or 3.
        """
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
        self.kernel_continuity_regularisation = kernel_continuity_regularisation

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
                    (
                        self.nb_in_channels
                        if scale_index == 0
                        else self.num_features[scale_index - 1]
                    ),
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
            nb_out = self.nb_channels

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
        """list of int: Kernel sizes for each dilated convolution scale."""
        if self._kernel_sizes is None:
            if self.spacetime_ndim == 2:
                self._kernel_sizes = [7, 5, 3, 3, 3, 3, 3, 3]
            elif self.spacetime_ndim == 3:
                self._kernel_sizes = [7, 5, 3, 3]

        return self._kernel_sizes

    @property
    def num_features(self):
        """list of int: Number of features for each dilated convolution scale."""
        if self._num_features is None:
            if self.spacetime_ndim == 2:
                self._num_features = [64, 32, 16, 8, 4, 2, 1, 1]
            elif self.spacetime_ndim == 3:
                self._num_features = [10, 8, 4, 2]

        return self._num_features

    def forward(self, x):
        """Run the forward pass through the JINet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(B, C, ...spatial_dims...)``.

        Returns
        -------
        torch.Tensor
            Denoised output tensor.
        """
        dilated_conv_list = []

        # Calculate dilated convolutions
        for index in range(len(self.kernel_sizes)):
            x = self.dilated_conv_functions[index](x)
            dilated_conv_list.append(x)

        # Concat the results
        x = torch.cat(dilated_conv_list, dim=1)

        # First kernel size one conv
        x = self.kernel_one_conv_functions[0](x)
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

    def enforce_blind_spot(self):
        """Zero out the center pixel weight to enforce J-invariance.

        Must be called after each optimization step to guarantee the
        blind-spot property required for self-supervised denoising.
        Sets the center kernel weight of the first dilated convolution
        layer to zero.
        """
        with torch.no_grad():
            first_conv = self.dilated_conv_functions[0].conv
            center = tuple((k - 1) // 2 for k in first_conv.kernel_size)
            if self.spacetime_ndim == 2:
                first_conv.weight[:, :, center[0], center[1]] = 0
            elif self.spacetime_ndim == 3:
                first_conv.weight[:, :, center[0], center[1], center[2]] = 0

    def post_optimisation(self):
        """Apply kernel smoothing regularization after each optimization step.

        When ``kernel_continuity_regularisation`` is enabled, smooths
        the dilated convolution kernels by convolving them with a small
        averaging filter. The smoothing strength decreases for deeper layers.
        """
        if not self.kernel_continuity_regularisation:
            return

        b = 0.0005
        with torch.no_grad():
            for dilated_conv_module in self.dilated_conv_functions:
                weights = dilated_conv_module.conv.weight
                num_channels = weights.shape[1]

                if self.spacetime_ndim == 2:
                    kernel = numpy.array(
                        [[b, b, b], [b, 1, b], [b, b, b]], dtype=numpy.float32
                    )
                    kernel = kernel[numpy.newaxis, numpy.newaxis, ...]
                    kernel_t = torch.from_numpy(kernel).to(weights.device)
                    kernel_t = kernel_t / kernel_t.sum()
                    kernel_t = kernel_t.expand(num_channels, 1, -1, -1)
                    new_weights = F.conv2d(
                        weights, kernel_t, groups=num_channels, padding=1
                    )
                elif self.spacetime_ndim == 3:
                    kernel = numpy.full((3, 3, 3), b, dtype=numpy.float32)
                    kernel[1, 1, 1] = 1.0
                    kernel = kernel[numpy.newaxis, numpy.newaxis, ...]
                    kernel_t = torch.from_numpy(kernel).to(weights.device)
                    kernel_t = kernel_t / kernel_t.sum()
                    kernel_t = kernel_t.expand(num_channels, 1, -1, -1, -1)
                    new_weights = F.conv3d(
                        weights, kernel_t, groups=num_channels, padding=1
                    )

                dilated_conv_module.conv.weight.data.copy_(new_weights)

                # Decrease smoothing for deeper layers
                b *= 0.5

    def fill_blind_spot(self):
        """Interpolate the blind-spot center weight after training.

        Must be called after training to fill in the zeroed center
        pixel weight. Uses a neighbor-average filter to estimate the
        missing center weight, then rescales the kernel to preserve
        the original weight sum.
        """
        with torch.no_grad():
            first_conv = self.dilated_conv_functions[0].conv
            weights = first_conv.weight
            num_channels = weights.shape[1]
            out_ch = weights.shape[0]

            # Save per-output-channel sums before modification
            original_sums = [weights[oc].sum().item() for oc in range(out_ch)]

            if self.spacetime_ndim == 2:
                b = 1.0
                kernel = numpy.array(
                    [[b, b, b], [b, 0, b], [b, b, b]], dtype=numpy.float32
                )
                kernel = kernel[numpy.newaxis, numpy.newaxis, ...]
                kernel_t = torch.from_numpy(kernel).to(weights.device)
                kernel_t = kernel_t / kernel_t.sum()
                kernel_t = kernel_t.expand(num_channels, 1, -1, -1)
                filtered = F.conv2d(weights, kernel_t, groups=num_channels, padding=1)

                center = tuple((k - 1) // 2 for k in first_conv.kernel_size)
                weights[:, :, center[0], center[1]] = filtered[
                    :, :, center[0], center[1]
                ]

            elif self.spacetime_ndim == 3:
                b = 1.0
                kernel = numpy.full((3, 3, 3), b, dtype=numpy.float32)
                kernel[1, 1, 1] = 0.0
                kernel = kernel[numpy.newaxis, numpy.newaxis, ...]
                kernel_t = torch.from_numpy(kernel).to(weights.device)
                kernel_t = kernel_t / kernel_t.sum()
                kernel_t = kernel_t.expand(num_channels, 1, -1, -1, -1)
                filtered = F.conv3d(weights, kernel_t, groups=num_channels, padding=1)

                center = tuple((k - 1) // 2 for k in first_conv.kernel_size)
                weights[:, :, center[0], center[1], center[2]] = filtered[
                    :, :, center[0], center[1], center[2]
                ]

            # Rescale per output channel to preserve per-channel weight sums
            for oc in range(out_ch):
                new_sum = weights[oc].sum().item()
                if abs(new_sum) > 1e-10:
                    weights[oc] *= original_sums[oc] / new_sum

            first_conv.weight.data.copy_(weights)
