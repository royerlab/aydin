from abc import ABC

import numpy


class ShapeNormaliser(ABC):
    """Shape Normaliser

    Parameters
    ----------
    clip : bool
    epsilon : float
    shape_normalisation : bool
    transform : str

    """

    epsilon: float
    leave_as_float: bool
    clip: bool
    original_dtype: numpy.dtype

    def __init__(self, batch_axes=None, channel_axes=None):
        """Constructs a normalisers"""
        self.batch_axes = batch_axes
        self.channel_axes = channel_axes
        self.axis_permutation = None
        self.permutated_image_shape = None

    def normalise(self, array):
        """Normalises the given array in-place (if possible).

        Parameters
        ----------
        array : numpy.ndarray
            array to normalisers
        batch_dims : list
        channel_dims : list

        Returns
        -------
        array : numpy.ndarray

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
        """Denormalises the given array in-place (if possible).

        Parameters
        ----------
        array : numpy.ndarray

        Returns
        -------
        array : numpy.ndarray

        """

        array = self.shape_denormalize(
            array,
            axes_permutation=self.axis_permutation,
            permutated_image_shape=self.permutated_image_shape,
        )

        return array

    @staticmethod
    def shape_normalize(image, batch_axes=None, channel_axes=None):
        """Permutates batch dimensions to the front and collapse into
        one dimension. Resulting array has to be in the form of (B,...)
        where B is the number of batch dimensions.

        Parameters
        ----------
        image : numpy.ndarray
        batch_axes : list
        channel_axes : list

        Returns
        -------
        Tuple of normalized_image, axes_permutation, permutated_image.shape : tuple

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
        """Denormalizes the shape of an image from normalized form to the
        original image form.

        Parameters
        ----------
        image : numpy.ndarray
        axes_permutation : array_like
        permutated_image_shape : tuple

        Returns
        -------
        array : numpy.ndarray

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
