from typing import Optional

import numpy
from numpy.typing import ArrayLike
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import medfilt2d

from aydin.it.classic_denoisers import _defaults
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariance import calibrate_denoiser


def calibrate_denoise_gm(
    image: ArrayLike,
    max_filter_size: int = 3,
    crop_size_in_voxels: Optional[int] = _defaults.default_crop_size_normal,
    optimiser: str = _defaults.default_optimiser,
    max_num_evaluations: int = _defaults.default_max_evals_normal,
    enable_extended_blind_spot: bool = True,
    display_images: bool = False,
    display_crop: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates the Gaussian-Median mix denoiser for the given image and returns
    the optimal parameters obtained using the N2S loss.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate denoiser for.

    max_filter_size : int
        Max filter size to use during calibration.
        Should be a positive odd number such as 3, 5, 7, ...

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate denoiser. Increase this
        number by factors of two if denoising quality is unsatisfactory.

    optimiser: str
        Optimiser to use for finding the best denoising
        parameters. Can be: 'smart' (default), or 'fast' for a mix of SHGO
        followed by L-BFGS-B.
        (advanced)

    max_num_evaluations: int
        Maximum number of evaluations for finding the optimal parameters.
        Increase this number by factors of two if denoising quality is
        unsatisfactory.

    enable_extended_blind_spot: bool
        Set to True to enable extended blind-spot detection.
        (advanced)

    display_images: bool
        When True the denoised images encountered during optimisation are shown

    display_crop: bool
        Displays crop, for debugging purposes...
        (advanced)

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
    crop = representative_crop(
        image, crop_size=crop_size_in_voxels, display_crop=display_crop
    )

    # Size range:
    filter_size_range = [3, 5, 7]

    # filter sizes:
    filter_size_range = list((s for s in filter_size_range if s <= max_filter_size))

    # Sigma range:
    sigma_range = (0.01, 3.0)

    # factor range:
    factor_range = (1.0, 3.0)

    # Alpha range:
    mixing_range = (0.0, 1.0)

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {
        'size': filter_size_range,
        'sigma': sigma_range,
        'factor': factor_range,
        'alpha': mixing_range,
        'beta': mixing_range,
        'gamma': mixing_range,
    }

    # Calibrate denoiser
    best_parameters = (
        calibrate_denoiser(
            crop,
            denoise_gm,
            mode=optimiser,
            denoise_parameters=parameter_ranges,
            max_num_evaluations=max_num_evaluations,
            blind_spots=enable_extended_blind_spot,
            display_images=display_images,
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = 3 * image.nbytes

    return denoise_gm, best_parameters, memory_needed


def denoise_gm(
    image: ArrayLike,
    sigma: float = 0.5,
    size: int = 3,
    factor: float = 2,
    alpha: float = 0.25,
    beta: float = 0.3,
    gamma: float = 0.5,
    **kwargs,
):
    """
    Denoises the given image with a linear mixing of median
    filtering and Gaussian filtering. Simple and fast but quite
    effective for low to moderate noise levels and images with
    band-limited signal (~ there is a 'PSF').
    \n\n
    Note: We recommend applying a variance stabilisation transform
    to improve results for images with non-Gaussian noise.

    Parameters
    ----------
    image: ArrayLike
        Image to be denoised.

    sigma: float
        Gaussian blur sigma.

    size: int
        Size of the median filter

    factor: float
        Ratio between the scales of the two scales

    alpha: float
        First mixing coefficient.

    beta: float
        First mixing coefficient.

    gamma: float
        First mixing coefficient.


    Returns
    -------
    Denoised image
    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    size_1 = size
    size_2 = int(factor * size / 2) * 2 + 1
    sigma_1 = sigma
    sigma_2 = factor * sigma

    wm1 = alpha * beta
    wm2 = alpha * (1 - beta)
    wg1 = (1 - alpha) * gamma
    wg2 = (1 - alpha) * (1 - gamma)

    if image.ndim == 2:
        denoised = wm1 * medfilt2d(image, kernel_size=size_1)
    else:
        denoised = wm1 * median_filter(image, size=size_1)

    if image.ndim == 2:
        denoised += wm2 * medfilt2d(image, kernel_size=size_2)
    else:
        denoised += wm2 * median_filter(image, size=size_2)

    denoised += wg1 * gaussian_filter(image, sigma=sigma_1)
    denoised += wg2 * gaussian_filter(image, sigma=sigma_2)

    return denoised
