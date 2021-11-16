from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import (
    Concatenate,
    Conv2D,
    Conv3D,
    LeakyReLU,
    Activation,
)
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.models import Model

from aydin.nn.models.utils.conv_block import conv2d_torch, conv3d_torch
from aydin.nn.models.utils.training_architectures import get_jinet_fit_args


class JINetModel(Model):
    """JINet model for 2D and 3D images."""

    def __init__(
        self,
        input_layer_size,
        spacetime_ndim,
        num_output_channels: int = 1,
        kernel_sizes: int = None,
        num_features: int = None,
        num_dense_layers: int = 3,
        num_channels: int = None,
        final_relu: bool = False,
        degressive_residuals: bool = True,
        learning_rate: float = 0.01,
        **kwargs,
    ):
        """

        Parameters
        ----------
        input_layer_size
        spacetime_ndim
        num_output_channels : int
            number of output channels
        kernel_sizes : int
            a list of kernel sizes; corresponding to num_features
        num_features : int
            a list of number of channels; corresponding to kernel_sizes
        num_dense_layers : int
            number of dense layers after feature extraction
        num_channels : int
            number of channels in the dense layer
        final_relu : bool
            whether having the final ReLU or not
        degressive_residuals : bool
            whether having weight decay in the dense layers
        learning_rate : float
        kwargs
        """
        if spacetime_ndim != 2 and spacetime_ndim != 3:
            raise Exception("Currently only JINet2D and JINet3D is supported.")

        self.spacetime_ndim = spacetime_ndim
        self.num_dense_layers = num_dense_layers
        self.num_channels = num_channels
        self.num_output_channels = num_output_channels
        self.final_relu = final_relu
        self.degressive_residuals = degressive_residuals

        if type(input_layer_size) is int:
            input_layer_size = (input_layer_size,) * self.spacetime_ndim + (1,)

        # These are the scales and associated kernel sizes and number of features
        if kernel_sizes is None:
            if self.spacetime_ndim == 2:
                kernel_sizes = [7, 5, 3, 3, 3, 3, 3, 3]
            elif self.spacetime_ndim == 3:
                kernel_sizes = [7, 5, 3, 3]
        if num_features is None:
            if self.spacetime_ndim == 2:
                num_features = [64, 32, 16, 8, 4, 2, 1, 1]
            elif self.spacetime_ndim == 3:
                num_features = [10, 8, 4, 2]
        self.kernel_sizes = kernel_sizes
        self.num_features = num_features
        if len(kernel_sizes) != len(num_features):
            raise ValueError("Number of kernel sizes and features does not match.")

        # Construct a model
        input_lyr = Input(input_layer_size, name='input')
        y = self.jinet_core(input_lyr)
        super().__init__(input_lyr, y)

        # Compile the model
        self.compile(optimizer=Adam(lr=learning_rate), loss='mse')
        self.compiled = True

    def size(self):
        """Returns size of the model in bytes"""
        return self.count_params() * 4

    def fit(
        self,
        input_image,
        target_image,
        batch_size,
        callbacks,
        verbose=None,
        max_epochs=None,
        total_num_patches=None,
        img_val=None,
        create_patches_for_validation=None,
        train_valid_ratio=None,
    ):
        """

        Parameters
        ----------
        input_image
        target_image
        batch_size
        callbacks
        verbose
        max_epochs
        total_num_patches
        img_val
        create_patches_for_validation
        train_valid_ratio

        Returns
        -------
        loss_history

        """
        validation_data = get_jinet_fit_args(
            input_image,
            batch_size,
            total_num_patches,
            img_val,
            create_patches_for_validation,
            train_valid_ratio,
        )

        loss_history = super().fit(
            input_image,
            target_image,
            epochs=max_epochs,
            callbacks=callbacks,
            verbose=verbose,
            batch_size=batch_size,
            validation_data=validation_data,
        )

        return loss_history

    def predict(
        self,
        x,
        batch_size=None,
        verbose=0,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
        """Overwritten model predict method.

        Parameters
        ----------
        x
        batch_size
        verbose
        steps
        callbacks
        max_queue_size
        workers
        use_multiprocessing

        Returns
        -------

        """
        # TODO: move as much as you can from it cnn _translate
        return super().predict(
            x,
            batch_size=batch_size,
            verbose=verbose,
        )

    def jinet_core(self, input_lyr):
        dilated_conv_list = []
        total_num_features = 0
        current_receptive_field_radius = 0
        x = input_lyr
        for scale_index in range(len(self.kernel_sizes)):
            # Get kernel size and number of features:
            size = self.kernel_sizes[scale_index]
            num = self.num_features[scale_index]

            # radius and dilation:
            radius = (size - 1) // 2
            dilation = 1 + current_receptive_field_radius

            # Setup dilated convolution:
            if self.spacetime_ndim == 2:
                dilated_conv_method = conv2d_torch
            elif self.spacetime_ndim == 3:
                dilated_conv_method = conv3d_torch

            x = dilated_conv_method(
                x,  # current_num_channels,  # channel_in
                num,  # number of filters: channel_out
                kernel_size=size,
                padding=dilation * radius,
                dilation_rate=dilation,
                lyrname=f'dilcv{scale_index}',
                act='lrel',
                leaky_alpha=0.01,
            )

            # We keep track of the total number of features until now:
            total_num_features += num

            # we update the current receptive field radius:
            current_receptive_field_radius += dilation * radius

            # append convolution ton the list:
            dilated_conv_list.append(x)

        # stack all features into one tensor:
        x = Concatenate(axis=-1)(dilated_conv_list)

        # We keep the number of features:
        self.total_num_features = total_num_features

        # By default the number of channels for the deep layers is the total number of spatial features generated
        if self.num_channels is None:
            num_channels = 2 * total_num_features

        y = None
        f = 1
        for level_index in range(self.num_dense_layers):
            # number of input and output channels:
            # num_in = total_num_features if level == 0 else num_channels
            num_out = (
                self.num_output_channels
                if level_index == self.num_dense_layers - 1
                else num_channels
            )

            # Setup of channel-wise dense layer (1x1 convolution)
            if self.spacetime_ndim == 2:
                channel_wise_dense_layer_type = Conv2D
            elif self.spacetime_ndim == 3:
                channel_wise_dense_layer_type = Conv3D

            x = channel_wise_dense_layer_type(
                num_out, kernel_size=1, padding='valid', name=f'dense_cv{level_index}'
            )(x)
            x = LeakyReLU(alpha=0.01)(x)

            if y is None:
                y = x
            else:
                y = y + f * x

        y = channel_wise_dense_layer_type(
            self.num_output_channels, kernel_size=1, padding='same', name='final_cv'
        )(y)

        if self.final_relu:
            y = Activation('relu', name='final_relu')(y)
        return y
