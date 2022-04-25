from typing import Optional

import numpy
from numba import jit

from aydin.it.classic_denoisers.butterworth import calibrate_denoise_butterworth
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.log.log import lprint, lsection


def dimension_analysis_on_image(
    image,
    epsilon: float = 0.05,
    min_spatio_temporal: int = 2,
    max_spatio_temporal: int = 4,
    max_channels_per_axis: int = 0,
    crop_size_in_voxels: Optional[int] = 512000,
    crop_timeout_in_seconds: float = 5,
    max_num_evaluations: Optional[int] = 21,
):
    """
    Analyses an image and tries to determine which dimensions are
    spatio-temporal, batch, and channel dimensions. Spatio-temporal
    dimensions are dimensions that are not batch or channel dimensions. For
    the purpose of image denoising, the cardinal rule is to consider a
    dimension as spatio-temporal if there is a degree of continuity of the
    true underlying noiseless signal. So we determine the degree of
    correlation along each axis. Axis for which there is no (sufficient)
    correlation are assumed to be batch or channel dimensions. We guess which
    are channel dimensions on the basis of the number of putative channels per
    axis. By default we avoid guessing channels because it is rarely useful for
    the purpose of denoising.

    Parameters
    ----------
    image : ndarray
        Image to analyse.

    algorithm : str
       Algorithm used to analyse dimensions. May be: 'butterworth'.

    epsilon : float
        Value below which a correlation is considered zero.

    min_spatio_temporal: int
        Minimum number of spatio-temporal dimensions.

    max_spatio_temporal: int
        Maximum number of spatio-temporal dimensions.

    max_channels_per_axis: int
        Max number of channels per chanhnel axis.

    max_sigma : float
        Max sigma for correlation determination

    crop_size_in_voxels : int
        Size of crop in voxels, used to speed up computation.

    crop_timeout_in_seconds: int
        Time-out in seconds for finding the best crop to analyse dimensions.

    max_num_evaluations : int
        Maximum number of evaluations per axis.

    Returns
    -------
    List of batch axes and list of channel axes.



    """
    with lsection(
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
        index = min(min_spatio_temporal - 1, len(sorted_values) - 1)

        # This is the corresponding threshold so that at least these n dimensions are spatio-temporal:
        threshold = sorted_values[index]

        # But we have to make sure that
        threshold = max(1 - epsilon, threshold)

        lprint(
            f"Correlation values per axis: {values} (lower values means more correlation)"
        )

        lprint(
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

        lprint(
            f"Inferred batch axes: {batch_axes} and channel axes: {channel_axes} for image of shape {image.shape}"
        )

        return batch_axes, channel_axes


@jit(nopython=True, parallel=True)
def _normalise(image, min_value, max_value):
    image = numpy.clip(image, a_min=min_value, a_max=max_value)
    image = (image - min_value) / (max_value - min_value)
    return image
