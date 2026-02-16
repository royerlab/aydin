"""Shape normaliser for batch and channel dimension handling.

Provides `ShapeNormaliser` which normalizes image arrays into a canonical
shape of (B, C, *spatial_dims) by permuting and collapsing batch and channel
dimensions, and denormalizes back to the original shape.
"""

from abc import ABC

import numpy


class ShapeNormaliser(ABC):
    """Shape normaliser for batch and channel dimension handling.

    Normalizes image arrays into a canonical shape of
    (B, C, *spatial_dims) by permuting and collapsing batch and
    channel dimensions. Tracks the permutation to enable
    denormalization back to the original shape.
    """

    def __init__(self, batch_axes=None, channel_axes=None):
        """Construct a ShapeNormaliser.

        Parameters
        ----------
        batch_axes : tuple of bool, optional
            Boolean flags indicating which axes are batch dimensions.
        channel_axes : tuple of bool, optional
            Boolean flags indicating which axes are channel dimensions.
        """
        self.batch_axes = batch_axes
        self.channel_axes = channel_axes
        self.axis_permutation = None
        self.permutated_image_shape = None

    def normalise(self, array):
        """Normalize the array shape to (B, C, *spatial_dims) form.

        Permutes and collapses batch and channel dimensions to the front.

        Parameters
        ----------
        array : numpy.ndarray
            Array to normalize.

        Returns
        -------
        numpy.ndarray
            Shape-normalized array with shape (B, C, *spatial_dims).
        """
        (
            array,
            self.axis_permutation,
            self.permutated_image_shape,
        ) = self.shape_normalize(
            array, batch_axes=self.batch_axes, channel_axes=self.channel_axes
        )

        return array

    def denormalise(self, array: numpy.ndarray, **kwargs):
        """Restore the array to its original shape before normalization.

        Reverses the permutation and reshaping applied by `normalise`.

        Parameters
        ----------
        array : numpy.ndarray
            Shape-normalized array to denormalize.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        numpy.ndarray
            Array restored to its original dimension ordering.
        """

        array = self.shape_denormalize(
            array,
            axes_permutation=self.axis_permutation,
            permutated_image_shape=self.permutated_image_shape,
        )

        return array

    @staticmethod
    def shape_normalize(image, batch_axes=None, channel_axes=None):
        """Permute and collapse batch/channel dimensions to canonical form.

        Moves batch dimensions to the front and channel dimensions next,
        then collapses each group into a single dimension. Singleton
        non-channel dimensions are treated as batch dimensions.

        Parameters
        ----------
        image : numpy.ndarray
            Image array to normalize.
        batch_axes : tuple of bool, optional
            Boolean flags for batch axes. If None, no axes are batch.
        channel_axes : tuple of bool, optional
            Boolean flags for channel axes. If None, no axes are channels.

        Returns
        -------
        tuple of (numpy.ndarray, list of int, tuple of int)
            A tuple of (normalized_image, axes_permutation,
            permutated_image_shape) needed for denormalization.
        """
        if batch_axes is None:
            batch_axes = (False,) * len(image.shape)
        if channel_axes is None:
            channel_axes = (False,) * len(image.shape)

        # Singleton dimensions are automatically batch dimension, unless it is a channel dimension, trivially.
        batch_axes = tuple(
            True if s == 1 and not c else b
            for b, c, s in zip(batch_axes, channel_axes, image.shape)
        )

        # get indices for different types of dimensions and their length
        batch_indices = [index for index, value in enumerate(batch_axes) if value]
        batch_length = int(numpy.prod([image.shape[index] for index in batch_indices]))

        channel_indices = (
            [index for index, value in enumerate(channel_axes) if value]
            if channel_axes
            else []
        )
        channel_length = int(
            numpy.prod([image.shape[index] for index in channel_indices])
        )

        spacetime_indices = [
            index
            for index in range(len(image.shape))
            if index not in batch_indices + channel_indices
        ]

        # Axes permutation
        axes_permutation = batch_indices + channel_indices + spacetime_indices

        # Bring all batch dimensions to front
        permutated_image = numpy.transpose(image, axes_permutation)

        # Collapse batch dimensions into one, same for channel dimensions
        spacetime_shape = tuple([image.shape[i] for i in spacetime_indices])
        normalized_shape = (batch_length, channel_length) + spacetime_shape
        normalized_shape = tuple(
            s for i, s in enumerate(normalized_shape) if s != 1 or i <= 1
        )

        # Reshape array:
        normalized_image = permutated_image.reshape(normalized_shape)

        return (normalized_image, axes_permutation, permutated_image.shape)

    @staticmethod
    def shape_denormalize(image, axes_permutation, permutated_image_shape):
        """Restore an image from normalized shape to its original form.

        Parameters
        ----------
        image : numpy.ndarray
            Shape-normalized image to restore.
        axes_permutation : list of int
            Axes permutation from `shape_normalize`.
        permutated_image_shape : tuple of int
            Intermediate shape from `shape_normalize`.

        Returns
        -------
        numpy.ndarray
            Image restored to its original dimension ordering and shape.
        """
        spatiotemp_shape = image.shape[2:]

        # Number of spatio-temp dimensions:
        num_spatiotemp_dims = len(spatiotemp_shape)

        # If the input image has a different lengths along the spatio-temporal dimensions,
        # that's fine, we accommodate for it:
        # Note: that's only fine for spatio-temp dim, not batch or channels!
        adapted_permutated_image_shape = list(permutated_image_shape)
        adapted_permutated_image_shape[-num_spatiotemp_dims:] = spatiotemp_shape
        adapted_permutated_image_shape = tuple(adapted_permutated_image_shape)

        # Reshape the input to its permutated shape:
        permutated_image = image.reshape(adapted_permutated_image_shape)

        # Retrieves dimensions back:
        return numpy.transpose(permutated_image, axes=numpy.argsort(axes_permutation))
