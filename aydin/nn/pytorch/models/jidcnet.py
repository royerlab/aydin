import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    JIDCnet2D -- J-invariant Dilated Convolution Net
"""


class JInet2D(nn.Module):
    def __init__(
        self,
        num_input_channels=1,
        num_output_channels=1,
        kernel_sizes=None,
        num_features=None,
        num_dense_layers=3,
        num_channels=None,
        final_relu=False,
        full_convolutional_across_channels=True,
        padding_mode='zeros',
        degressive_residuals=True,
        kernel_continuity_regularisation=True,
        sine_activations=True,
    ):
        """

        Parameters
        ----------
        num_input_channels
        num_output_channels
        kernel_sizes
        num_features
        num_dense_layers
        num_channels
        final_relu
        """
        super().__init__()

        # These are the scales and associated kernel sizes and number of features

        if kernel_sizes is None:
            # kernel_sizes = [5, 5, 5, 3, 3, 3, 3, 3]
            kernel_sizes = [7, 5, 3, 3, 3, 3, 3, 3]
        if num_features is None:
            # num_features = [24, 24, 24, 12, 12, 12, 2, 1]
            num_features = [64, 32, 16, 8, 4, 2, 1, 1]
        self.kernel_sizes = kernel_sizes
        self.num_features = num_features
        assert len(kernel_sizes) == len(num_features)

        # How many scales?
        self.num_scales = len(kernel_sizes)

        # final relu?
        self.final_relu = final_relu

        # Degressive residuals ?
        self.degressive_residuals = degressive_residuals

        # Enforce kernel continuity?
        self.kernel_continuity_regularisation = kernel_continuity_regularisation

        # Sine activations?
        self.sine_activations = sine_activations

        self.dilated_conv_list = nn.ModuleList()
        total_num_features = 0
        current_receptive_field_radius = 0
        current_num_channels = num_input_channels
        for scale_index in range(self.num_scales):
            # Get kernel size and number of features:
            size = kernel_sizes[scale_index]
            num = num_features[scale_index]

            # radius and dilation:
            radius = (size - 1) // 2
            dilation = 1 + current_receptive_field_radius

            # Do we go full convolutional across channels or do we try to minimise parameters?
            num_groups = (
                1
                if full_convolutional_across_channels
                else min(current_num_channels, num)
            )

            # Setup dilated convolution:
            dilated_convolution = nn.Conv2d(
                current_num_channels,
                num,
                kernel_size=size,
                padding=dilation * radius,
                dilation=dilation,
                padding_mode=padding_mode,
                groups=num_groups,
            )
            print(dilated_convolution)

            # We keep track of the total number of features until now:
            total_num_features += num

            # we update the current receptive field radius:
            current_receptive_field_radius += dilation * radius

            # next convolution num of input channels is this last convolution number of output channels:
            current_num_channels = num

            # append convolution ton the list:
            self.dilated_conv_list.append(dilated_convolution)

        # we save the receptive field radius:
        self.receptive_field_radius = current_receptive_field_radius

        # We keep the number of features:
        self.total_num_features = total_num_features

        # By default the number of channels for the deep layers is the total number of spatial features generated
        if num_channels is None:
            num_channels = 2 * total_num_features

        # Instanciates dense layers:
        self.dense_layers_list = nn.ModuleList()
        for level in range(0, num_dense_layers):
            # number of input and output channels:
            num_in = total_num_features if level == 0 else num_channels
            num_out = (
                num_output_channels if level == num_dense_layers - 1 else num_channels
            )

            # Setup of channel-wise dense layer (1x1 convolution)
            dense = nn.Conv2d(num_in, num_out, kernel_size=1, padding=0)
            self.dense_layers_list.append(dense)

        self.finalconv = nn.Conv2d(
            num_channels, num_output_channels, kernel_size=1, padding=0
        )

    def forward(self, x0):
        """

        Parameters
        ----------
        x0  input

        Returns output of model
        -------

        """
        # Deep spatial feature generation:
        x = x0
        features = []
        for scale_index in range(self.num_scales):
            dilated_convolution = self.dilated_conv_list[scale_index]
            x = dilated_convolution(x)
            features.append(x)
            x = F.leaky_relu(x, negative_slope=0.01)

        # stack all features into one tensor:
        x = torch.cat(features, 1)

        # Residual 'feed-forward' perceptron for inference, i.e. 1x1 convolutions
        y = None
        f = 1
        for dense in self.dense_layers_list:
            x = dense(x)
            if self.sine_activations:
                x = torch.sin(x)
            else:
                x = F.leaky_relu(x, negative_slope=0.01)
            if y is None:
                y = x
            else:
                y = y + f * x

            if self.degressive_residuals:
                # deeper layers are expected to contribute less and less to solution:
                f *= 0.5

        # final dense layer to bring down number of channels:
        y = self.finalconv(y)

        # final relu:
        if self.final_relu:
            y = torch.clamp(y, 0, 1)

        return y

    def enforce_blind_spot(self):
        """
        This must be called after each optimisation step to guarantee J-invariance
        """

        with torch.no_grad():
            for skipconv in self.dilated_conv_list[0:1]:  #
                indexes = tuple((i - 1) // 2 for i in skipconv.kernel_size)
                skipconv._parameters['weight'][:, :, indexes[0], indexes[1]] = 0

    def post_optimisation(self):
        """
        This must be called after each optimisation for kernel regularisation
        """

        b = 0.0005
        with torch.no_grad():

            if self.kernel_continuity_regularisation:
                for skipconv in self.dilated_conv_list:  # [0:2]
                    weights = skipconv._parameters['weight']
                    num_channels = weights.shape[1]
                    kernel = numpy.array([[b, b, b], [b, 1, b], [b, b, b]])
                    kernel = kernel[numpy.newaxis, numpy.newaxis, ...].astype(
                        numpy.float32
                    )
                    kernel = torch.from_numpy(kernel).to(weights.device)
                    kernel /= kernel.sum()
                    kernel = kernel.expand(num_channels, 1, -1, -1)
                    weights = F.conv2d(weights, kernel, groups=num_channels, padding=1)
                    skipconv._parameters['weight'] = weights

                    # decrease for upper layers:
                    b *= 0.5

    def fill_blind_spot(self):
        """
        This must be called after training to remove the blind-spot.
        """

        for skipconv in self.dilated_conv_list[0:1]:
            weights = skipconv._parameters['weight']
            num_channels = weights.shape[1]
            original_sum = weights.sum()
            b = 1
            kernel = numpy.array([[b, b, b], [b, 0, b], [b, b, b]])
            kernel = kernel[numpy.newaxis, numpy.newaxis, ...].astype(numpy.float32)
            kernel = torch.from_numpy(kernel).to(weights.device)
            kernel /= kernel.sum()
            kernel = kernel.expand(num_channels, 1, -1, -1)
            filtered_weights = F.conv2d(weights, kernel, groups=num_channels, padding=1)

            indexes = tuple((i - 1) // 2 for i in skipconv.kernel_size)
            weights[:, :, indexes[0], indexes[1]] = filtered_weights[
                :, :, indexes[0], indexes[1]
            ]
            weights *= original_sum / weights.sum()

            skipconv._parameters['weight'] = weights

    def visualise_weights(self):
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            for i, dilated_conv in enumerate(self.dilated_conv_list):  #
                array = dilated_conv._parameters['weight'].cpu().detach().numpy()
                viewer.add_image(array, name=f'layer{i}', rgb=False)
