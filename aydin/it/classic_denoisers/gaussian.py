from functools import partial
from typing import Optional

import numpy
from scipy.ndimage import gaussian_filter

from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariant_smart import calibrate_denoiser_smart


def calibrate_denoise_gaussian(
    image,
    max_sigma: float = 2,
    max_num_truncate: int = 4,
    crop_size_in_voxels: Optional[int] = 128000,
    max_num_evaluations: int = 256,
    display_images: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates the Gaussian denoiser for the given image and returns the optimal
    parameters obtained using the N2S loss.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate denoiser for.

    max_sigma: float
        Maximum sigma for Gaussian filter.

    max_num_truncate: int
        Maximum number of Gaussian filter truncations to try.
        (advanced)

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate
        denoiser.
        (advanced)

    max_num_evaluations: int
        Maximum number of evaluations for finding
        the optimal parameters.
        (advanced)

    display_images: bool
        When True the denoised images encountered
        during optimisation are shown

    other_fixed_parameters: dict
        Any other fixed parameters

    Returns
    -------
    Denoising function, dictionary containing optimal parameters,
    and free memory needed in bytes for computation.
    """
    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # obtain representative crop, to speed things up...
    crop = representative_crop(image, crop_size=crop_size_in_voxels)

    # Size range:
    sigma_range = (0.0, max(0.0, max_sigma) + 1e-9)  # numpy.arange(0.2, 2, 0.1) ** 1.5

    # Truncate range (order matters: we want 4 -- the default -- first):
    truncate_range = [4, 8, 2, 1][: min(max_num_truncate, 4)]

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {'sigma': sigma_range, 'truncate': truncate_range}

    # Partial function:
    _denoise_gaussian = partial(denoise_gaussian, **other_fixed_parameters)

    # Calibrate denoiser
    best_parameters = (
        calibrate_denoiser_smart(
            crop,
            _denoise_gaussian,
            denoise_parameters=parameter_ranges,
            display_images=display_images,
            max_num_evaluations=max_num_evaluations,
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = 2 * image.nbytes

    return denoise_gaussian, best_parameters, memory_needed


def denoise_gaussian(image, sigma: float = 1, truncate: float = 4, **kwargs):
    """
    Denoises the given image using a simple Gaussian filter.
    Difficult to beat in terms of speed and often provides
    sufficient although not superb denoising performance. You
    should always try simple and fast denoisers first, and see
    if that works for you. If it works and is sufficient for
    your needs, why go for slower and more complex and slower
    approach? The only weakness of gaussian filtering is that it
    affects all frequencies. In contrast, the auto-tuned Butterworth
    denoiser will not blur within the estimated band-pass of
    the signal. Thus we recommend you use the Butterworth denoiser
    instead unless you have a good reason this use this one.
    \n\n
    Note: We recommend applying a variance stabilisation transform
    to improve results for images with non-Gaussian noise.

    Parameters
    ----------
    image: ArrayLike
        nD image to denoise

    sigma: float
        Standard deviation for Gaussian kernel.

    truncate: float
         Truncate the filter at this many standard deviations.

    Returns
    -------
    Denoised image
    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    return gaussian_filter(image, sigma=sigma, truncate=truncate)
