from functools import reduce
from math import ceil
import numpy


class BlindSpotTransform:
    """
    'BlindSpot' Transform (and inverse)

    Parameters
    ----------
    blind_spots : list

    """

    def __init__(self, blind_spots):
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
        """
        Get mask method

        Parameters
        ----------
        blind_spot_for_channel : tuple
        image_shape : tuple

        Returns
        -------
        Tuple of (mask, shape)

        """
        # First we prepare the mask:
        footprint = numpy.zeros(shape=self.footprint_shape, dtype=numpy.bool)
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
        """
        Transform method

        Parameters
        ----------
        input_image : numpy.ndarray
        target_image : numpy.ndarray

        Returns
        -------
        Tuple of (transformed_input_image, transformed_target_image) or (transformed_input_image, None)

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
        """
        Inverse transform method

        Parameters
        ----------
        translated_transformed_image : numpy.ndarray

        Returns
        -------
        translated_image : numpy.ndarray

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
