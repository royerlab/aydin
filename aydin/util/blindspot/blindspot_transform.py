"""Blind-spot transform for self-supervised image denoising.

Provides forward and inverse transforms that reorganize image pixels
according to blind-spot patterns, enabling Noise2Self-style training.
"""

from functools import reduce
from math import ceil

import numpy


class BlindSpotTransform:
    """Transform (and inverse) for blind-spot-based self-supervised denoising.

    Reorganizes image pixels according to a set of blind-spot offsets,
    producing batched multi-channel input/target pairs suitable for
    self-supervised training.

    Parameters
    ----------
    blind_spots : list of tuple of int
        List of coordinate offset tuples defining the blind-spot pattern.
        Each tuple specifies a relative pixel offset, e.g. [(0,0), (1,0), (-1,0)].
    """

    def __init__(self, blind_spots):
        """Initialize the blind-spot transform.

        Parameters
        ----------
        blind_spots : list of tuple of int
            List of coordinate offset tuples defining the blind-spot
            pattern. Each tuple specifies a relative pixel offset,
            e.g. ``[(0,0), (1,0), (-1,0)]``.
        """
        super().__init__()

        self.blind_spots = blind_spots

        self.min_bounds = reduce(
            (lambda tx, ty: tuple(min(x, y) for x, y in zip(tx, ty))), self.blind_spots
        )
        self.max_bounds = reduce(
            (lambda tx, ty: tuple(max(x, y) for x, y in zip(tx, ty))), self.blind_spots
        )
        self.footprint_shape = tuple(
            ma - mi + 1 for mi, ma in zip(self.min_bounds, self.max_bounds)
        )

    def get_mask(self, blind_spot_for_channel, image_shape):
        """Generate a tiled boolean mask for a single blind-spot channel.

        Parameters
        ----------
        blind_spot_for_channel : tuple of int
            Coordinate offset for the specific blind-spot channel.
        image_shape : tuple of int
            Shape of the image to generate the mask for.

        Returns
        -------
        mask : numpy.ndarray
            Boolean mask array at least as large as the image.
        shape : tuple of int
            Shape of the downsampled result after masking.
        """
        # First we prepare the mask:
        footprint = numpy.zeros(shape=self.footprint_shape, dtype=numpy.bool_)
        coordinates = tuple(
            int(x - m) for x, m in zip(blind_spot_for_channel, self.min_bounds)
        )
        footprint[coordinates] = True
        shape = tuple(
            int(ceil(s / fs)) for fs, s in zip(self.footprint_shape, image_shape)
        )
        mask = numpy.tile(footprint, reps=shape)
        return mask, shape

    def transform(self, input_image, target_image):
        """Apply the blind-spot transform to input and target images.

        Produces batched, multi-channel arrays where each batch element
        corresponds to a blind-spot offset and each channel to a shifted
        view of the input.

        Parameters
        ----------
        input_image : numpy.ndarray
            Input image array.
        target_image : numpy.ndarray or None
            Target image array, or None for inference mode.

        Returns
        -------
        transformed_input_image : numpy.ndarray
            Transformed input with shape (n_spots, n_spots, *downsampled_shape).
        transformed_target_image : numpy.ndarray or None
            Transformed target with shape (n_spots, 1, *downsampled_shape),
            or None if ``target_image`` is None.
        """

        input_batch_list = []
        target_batch_list = []
        for blind_spot_for_batch in self.blind_spots:

            shift_for_roll = tuple(-x for x in blind_spot_for_batch)

            # prepare input image:
            shifted_input_image = numpy.roll(input_image, shift=shift_for_roll)
            input_channels_list = []
            for blind_spot_for_channel in self.blind_spots:

                mask, shape = self.get_mask(blind_spot_for_channel, input_image.shape)

                # Then, we extract values:
                input_channel = shifted_input_image[mask].reshape(shape)
                input_channels_list.append(input_channel)

            input_batch = numpy.stack(input_channels_list)
            input_batch_list.append(input_batch)

            # Prepare target image:
            if target_image is not None:
                shifted_target_image = numpy.roll(target_image, shift=shift_for_roll)
                mask, shape = self.get_mask(
                    (0,) * target_image.ndim, shifted_target_image.shape
                )

                # Then, we extract values:
                target_channel = shifted_target_image[mask].reshape(shape)
                target_batch = target_channel[numpy.newaxis, ...]
                target_batch_list.append(target_batch)

        transformed_input_image = numpy.stack(input_batch_list)
        if target_image is not None:
            transformed_target_image = numpy.stack(target_batch_list)

            # We keep the target image shape and dtype handy for the inverse transform:
            self.target_image_shape = target_image.shape
            self.target_image_dtype = target_image.dtype

            return transformed_input_image, transformed_target_image

        else:
            return transformed_input_image, None

    def inverse_transform(self, translated_transformed_image):
        """Apply the inverse blind-spot transform to reconstruct the image.

        Reassembles a full image from the batched, blind-spot-transformed
        representation produced by a denoising model.

        Parameters
        ----------
        translated_transformed_image : numpy.ndarray
            Transformed and translated image array with shape
            (n_spots, 1, *downsampled_shape).

        Returns
        -------
        numpy.ndarray
            Reconstructed image with the original shape and dtype.
        """

        translated_image = numpy.empty(
            shape=self.target_image_shape, dtype=self.target_image_dtype
        )

        for batch_image, blind_spot in zip(
            translated_transformed_image, self.blind_spots
        ):
            mask, shape = self.get_mask(blind_spot, translated_image.shape)
            translated_image[mask] = batch_image.ravel()

        return translated_image
