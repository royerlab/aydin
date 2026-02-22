"""Pooling downsampling layer supporting average and max pooling.

Provides a configurable spatial downsampling module for 2D and 3D inputs
used in encoder paths of UNet architectures.
"""

from torch import nn


class PoolingDown(nn.Module):
    """Spatial downsampling via average or max pooling.

    Reduces spatial dimensions by a factor of 2 using either average
    or max pooling. Supports both 2D and 3D inputs.

    Parameters
    ----------
    spacetime_ndim : int
        Number of spatial dimensions (2 or 3).
    pooling_mode : str
        Pooling mode: ``'ave'`` for average pooling or ``'max'``
        for max pooling.
    """

    def __init__(self, spacetime_ndim, pooling_mode):
        """Initialize the PoolingDown layer.

        Parameters
        ----------
        spacetime_ndim : int
            Number of spatial dimensions (2 or 3).
        pooling_mode : str
            Pooling mode: ``'ave'`` for average pooling or ``'max'``
            for max pooling.
        """
        super(PoolingDown, self).__init__()

        self.spacetime_ndim = spacetime_ndim
        self.pooling_mode = pooling_mode
        if pooling_mode not in ('ave', 'max'):
            raise ValueError(
                f'pooling_mode must be "ave" or "max", got "{pooling_mode}"'
            )

        if spacetime_ndim == 2:
            self.average_pooling = nn.AvgPool2d((2, 2))
            self.max_pooling = nn.MaxPool2d((2, 2))
        else:
            self.average_pooling = nn.AvgPool3d((2, 2, 2))
            self.max_pooling = nn.MaxPool3d((2, 2, 2))

    def forward(self, x):
        """Downsample the input tensor by a factor of 2.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Spatially downsampled tensor.
        """

        if self.pooling_mode == 'ave':
            x = self.average_pooling(x)
        elif self.pooling_mode == 'max':
            x = self.max_pooling(x)

        return x
