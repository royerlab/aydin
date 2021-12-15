from typing import Optional

import numpy
from joblib import delayed, Parallel

from aydin.it.classic_denoisers.gaussian import calibrate_denoise_gaussian
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.log.log import lprint, lsection


def dimension_analysis_on_image(
    image,
    epsilon: float = 1e-3,
    max_channels_per_axis: int = 0,
    max_sigma: float = 16.0,
    crop_size_in_voxels: int = 200000,
    max_num_evaluations: int = 16,
    backend: Optional[str] = None,
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

    epsilon : float
        Value below which a correlation is considered zero.

    max_channels_per_axis: int
        Max number of channels per chanhnel axis.

    max_sigma : float
        Max sigma for correlation determination

    crop_size_in_voxels : int
        Size of crop in voxels, used to speed up computation.

    max_num_evaluations : int
        Maximum number of evaluations per axis.

    backend : str
        Backend used for computation.

    Returns
    -------
    List of batch axes and list of channel axes.



    """
    with lsection("Analysing dimensions"):

        # obtain representative crop, to speed things up...
        crop = representative_crop(
            image, crop_size=crop_size_in_voxels, max_time_in_seconds=0.1
        )

        crop = crop.astype(dtype=numpy.float32)

        # Number of dimensions:
        nb_dim = len(crop.shape)

        # Function to run per dimension:
        def compute_value(_image, _axis: int):
            _, best_parameters, _ = calibrate_denoise_gaussian(
                _image,
                axes=(_axis,),
                min_sigma=0.0,
                max_sigma=max_sigma,
                max_num_truncate=1,
                max_num_evaluations=max_num_evaluations,
            )

            value = best_parameters['sigma']
            return value

        # Run in paralell each axis:
        values = Parallel(backend=backend, n_jobs=-1)(
            delayed(compute_value)(crop, axis) for axis in range(nb_dim)
        )

        lprint(
            f"Correlation Values per axis: {values} (higher values means more correlation)"
        )

        # Batch axes:
        batch_axes = tuple(
            axis
            for axis in range(image.ndim)
            if values[axis] < epsilon and image.shape[axis] > max_channels_per_axis
        )

        # Channel axis are
        channel_axes = tuple(
            axis
            for axis in batch_axes
            if values[axis] >= epsilon and image.shape[axis] <= max_channels_per_axis
        )

        lprint(
            f"Inferred batch axes: {batch_axes} and channel axes: {channel_axes} for image of shape {image.shape}"
        )

        return batch_axes, channel_axes
