from torch import cat
from torch import nn


class JINetModel(nn.Module):
    def __init__(self):
        super(JINetModel, self).__init__()

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

    def forward(self, x):
        dilated_conv_list = []
        total_nb_features = 0
        current_receptive_field_radius = 0

        for scale_index in range(len(self.kernel_sizes)):
            # Get kernel size and number of features:
            kernel_size = self.kernel_sizes[scale_index]
            nb_features = self.num_features[scale_index]

            # radius and dilation:
            radius = (kernel_size - 1) // 2
            dilation = 1 + current_receptive_field_radius

            x = dilated_conv(x)

            total_nb_features += nb_features  # update the number of features until now

            current_receptive_field_radius += dilation * radius  # update receptive field radius

            dilated_conv_list.append(x)

        x = cat()(dilated_conv_list)  #  TODO: pass axis as -1
        
        if self.nb_channels is None:
            nb_channels = total_nb_features * 2
            
        y = None
        f = 1
        for level_index in range(self.nb_dense_layers):
            nb_out = self.nb_out_channels if level_index == (self.nb_dense_layers - 1) else nb_channels

            x = channelwise_dense_layer(x)  #  TODO: pass correct parameters
            x = nn.LeakyReLU(negative_slope=0.01)(x)

            y = x if y is None else y + f * x

        y = channelwise_dense_layer(y)  #  TODO: pass correct parameters

        if self.final_relu:
            y = nn.ReLU()(y)

        return y

def n2t_jinet_train_loop():
    writer = SummaryWriter()

    optimizer = ESAdam(
        chain(model.parameters()),
        lr=learning_rate,
        start_noise_level=training_noise,
        weight_decay=l2_weight_regularisation,
    )


    writer.flush()
    writer.close()


def n2s_jinet_train_loop():
    writer = SummaryWriter()

    optimizer = ESAdam(
        chain(model.parameters()),
        lr=learning_rate,
        start_noise_level=training_noise,
        weight_decay=l2_weight_regularisation,
    )


    writer.flush()
    writer.close()
