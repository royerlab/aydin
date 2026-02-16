"""Dimension analysis for n-dimensional images.

This module provides functions to automatically determine which dimensions of
an image are spatio-temporal (correlated), batch, or channel dimensions.
This is critical for correctly setting up denoising pipelines.
"""

from typing import Optional

import numpy
from numba import jit

from aydin.it.classic_denoisers.butterworth import calibrate_denoise_butterworth
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.log.log import aprint, asection


def dimension_analysis_on_image(
    image,
    epsilon: float = 0.05,
    min_spatio_temporal: int = 2,
    max_spatio_temporal: int = 4,
    max_channels_per_axis: int = 4,
    crop_size_in_voxels: Optional[int] = 512000,
    crop_timeout_in_seconds: float = 5,
    max_num_evaluations: Optional[int] = 21,
):
    """Analyze image dimensions to classify them as spatio-temporal, batch, or channel.

    Determines which dimensions of an n-dimensional image are spatio-temporal
    (i.e., exhibit signal continuity), batch, or channel dimensions. This is
    done by measuring the degree of correlation along each axis using
    Butterworth filtering. Axes with insufficient correlation are classified
    as batch or channel dimensions based on their length.

    Parameters
    ----------
    image : numpy.ndarray
        Image to analyze.

    epsilon : float
        Value below which a frequency cutoff is considered to indicate
        no meaningful correlation (dimension is not spatio-temporal).

    min_spatio_temporal : int
        Minimum number of spatio-temporal dimensions to enforce.

    max_spatio_temporal : int
        Maximum number of spatio-temporal dimensions to allow.

    max_channels_per_axis : int
        Maximum number of entries along an axis for it to be considered
        a channel axis (rather than a batch axis).

    crop_size_in_voxels : int, optional
        Size of the representative crop in voxels, used to speed up computation.

    crop_timeout_in_seconds : float
        Timeout in seconds for finding the best crop to analyze dimensions.

    max_num_evaluations : int, optional
        Maximum number of Butterworth filter evaluations per axis.

    Returns
    -------
    batch_axes : tuple of int
        Indices of axes identified as batch dimensions.
    channel_axes : tuple of int
        Indices of axes identified as channel dimensions.
    """
    with asection(
        "Analysing dimensions to determine which should be spatio-temporal, batch, or channel"
    ):

        # pre-crop so we can control the time it takes:
        image = representative_crop(
            image,
            crop_size=crop_size_in_voxels,
            timeout_in_seconds=crop_timeout_in_seconds,
        )

        # Robust normalisation of image:
        min_value = numpy.percentile(image, q=2)
        max_value = numpy.percentile(image, q=98)
        if max_value > min_value:
            image = _normalise(image, min_value, max_value)

        # Adjust min number of spatio-temporal dimensions:
        min_spatio_temporal = min(min_spatio_temporal, image.ndim)

        # Number of dimensions:
        nb_dim = len(image.shape)

        values = []
        for axis in range(nb_dim):
            _, best_parameters, _ = calibrate_denoise_butterworth(
                image,
                axes=(axis,),
                min_order=4,
                max_order=4,
                optimiser='smart',
                max_num_evaluations=max_num_evaluations,
                jinv_interpolation_mode='gaussian',
                # blind_spots=False,
                crop_size_in_voxels=crop_size_in_voxels,
                display_images=False,
            )

            value = best_parameters['freq_cutoff']
            values.append(value)

        # Let's ensure there is a minimum number of spatio-temporal dimensions:
        sorted_values = list(values)

        # Values very close to 1.0 are probably just that, 1.0:
        sorted_values = list([(1.0 if abs(1 - v) < 0.01 else v) for v in sorted_values])

        # We sort the values:
        sorted_values.sort()

        # This is the index for the top n (n=min_spatio_temporal) most correlated dimensions:

        threshold = 1 - epsilon
        for _ in range(len(sorted_values) - 1):
            absolute_gap = sorted_values[_ + 1] - sorted_values[_]
            if absolute_gap > 0.3 and 2 * sorted_values[_] < sorted_values[_ + 1]:
                threshold = sorted_values[_]
                break

        aprint(
            f"Correlation values per axis: {values} (lower values means more correlation)"
        )

        aprint(
            f"Threshold for identifying spatio-temporal dimension is ...<={threshold}"
        )

        # spatio-temporal axes:
        st_axes = tuple(axis for axis in range(nb_dim) if values[axis] <= threshold)

        # We ensure that there is enough spatio-temporal dimensions by adding dimensions from the 'end':
        for axis in reversed(range(nb_dim)):
            if len(st_axes) < min_spatio_temporal:
                st_axes = tuple(set(st_axes + (axis,)))

        # We ensure that there is not too many spatio-temporal dimensions by removing dimensions from the 'front':
        for axis in reversed(range(nb_dim)):
            if len(st_axes) > max_spatio_temporal:
                st_axes = tuple(a for a in st_axes if a != axis)

        # Batch axes:
        batch_axes = tuple(
            axis
            for axis in range(nb_dim)
            if axis not in st_axes and image.shape[axis] > max_channels_per_axis
        )

        # Channel axis are
        channel_axes = tuple(
            axis
            for axis in range(nb_dim)
            if axis not in st_axes and image.shape[axis] <= max_channels_per_axis
        )

        aprint(
            f"Inferred batch axes: {batch_axes} and channel axes: {channel_axes} for image of shape {image.shape}"
        )

        return batch_axes, channel_axes


@jit(nopython=True, parallel=True)
def _normalise(image, min_value, max_value):
    """Clip and normalize an image to the [0, 1] range.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array.
    min_value : float
        Lower clipping bound (maps to 0).
    max_value : float
        Upper clipping bound (maps to 1).

    Returns
    -------
    image : numpy.ndarray
        Normalized image in the [0, 1] range.
    """
    image = numpy.clip(image, a_min=min_value, a_max=max_value)
    image = (image - min_value) / (max_value - min_value)
    return image
