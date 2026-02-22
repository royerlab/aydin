"""DnCNN (Denoising Convolutional Neural Network) model architecture.

Implements the DnCNN architecture from Zhang et al. (2017) with support
for both 2D and 3D spatial inputs. A feed-forward denoising network
with batch normalization.
"""

from torch import nn


class DnCNNModel(nn.Module):
    """DnCNN denoising network with 2D and 3D support.

    Feed-forward CNN with batch normalization for image denoising,
    based on the architecture from Zhang et al. (2017). The network
    consists of an initial convolution + ReLU, followed by repeated
    convolution + batch normalization + ReLU blocks, and a final
    convolution layer. No bias is used in any convolution layer.
    <notgui>
    """

    def __init__(self, spacetime_ndim, num_of_layers=17, features=64):
        """Initialize the DnCNN model.

        Parameters
        ----------
        spacetime_ndim : int
            Number of spatial dimensions (2 or 3).
        num_of_layers : int
            Total number of convolution layers.
        features : int
            Number of intermediate feature channels.

        Raises
        ------
        ValueError
            If ``spacetime_ndim`` is not 2 or 3.
        """
        super().__init__()

        self.spacetime_ndim = spacetime_ndim
        self.nb_unet_levels = 0  # No pooling, so no alignment padding needed

        if spacetime_ndim == 2:
            conv_class = nn.Conv2d
            bn_class = nn.BatchNorm2d
        elif spacetime_ndim == 3:
            conv_class = nn.Conv3d
            bn_class = nn.BatchNorm3d
        else:
            raise ValueError("spacetime_ndim must be 2 or 3")

        kernel_size = 3
        padding = 1

        layers = []
        # First layer: Conv + ReLU (no BN, no bias)
        layers.append(conv_class(1, features, kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: Conv + BN + ReLU
        for _ in range(num_of_layers - 2):
            layers.append(
                conv_class(features, features, kernel_size, padding=padding, bias=False)
            )
            layers.append(bn_class(features))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: Conv only (no BN, no bias)
        layers.append(conv_class(features, 1, kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        """Run the forward pass through the DnCNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(B, 1, ...spatial_dims...)``.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as input.
        """
        return self.dncnn(x)
