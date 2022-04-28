from functools import partial
from typing import Optional, Tuple, List

import numpy
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter

from aydin.it.classic_denoisers import _defaults
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariance import calibrate_denoiser


def calibrate_denoise_gaussian(
    image: ArrayLike,
    axes: Optional[Tuple[int, ...]] = None,
    min_sigma: float = 1e-6,
    max_sigma: float = 2.0,
    crop_size_in_voxels: Optional[int] = _defaults.default_crop_size_large.value,
    optimiser: str = 'fast',
    max_num_evaluations: int = _defaults.default_max_evals_high.value,
    blind_spots: Optional[List[Tuple[int]]] = _defaults.default_blind_spots.value,
    jinv_interpolation_mode: str = 'median',
    display_images: bool = False,
    display_crop: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates the Gaussian denoiser for the given image and returns the optimal
    parameters obtained using the N2S loss.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate denoiser for.

    axes: Optional[Tuple[int,...]]
        Axes over which to apply low-pass filtering.
        (advanced)

    min_sigma: float
        Minimum sigma for Gaussian filter.
        (advanced)

    max_sigma: float
        Maximum sigma for Gaussian filter.

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate denoiser.
        Increase this number by factors of two if denoising quality is
        unsatisfactory -- this can be important for very noisy images.
        Values to try are: 65000, 128000, 256000, 320000.
        We do not recommend values higher than 512000.

    optimiser: str
        Optimiser to use for finding the best denoising
        parameters. Can be: 'smart' (default), or 'fast' for a mix of SHGO
        followed by L-BFGS-B.
        (advanced)

    max_num_evaluations: int
        Maximum number of evaluations for finding
        the optimal parameters.
        Increase this number by factors of two if denoising quality is
        unsatisfactory.

    blind_spots: bool
        List of voxel coordinates (relative to receptive field center) to
        be included in the blind-spot. For example, you can give a list of
        3 tuples: [(0,0,0), (0,1,0), (0,-1,0)] to extend the blind spot
        to cover voxels of relative coordinates: (0,0,0),(0,1,0), and (0,-1,0)
        (advanced) (hidden)

    jinv_interpolation_mode: str
        J-invariance interpolation mode for masking. Can be: 'median' or
        'gaussian'.
        (advanced)

    display_images: bool
        When True the denoised images encountered
        during optimisation are shown
        (advanced) (hidden)

    display_crop: bool
        Displays crop, for debugging purposes...
        (advanced) (hidden)

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

    # Default axes:
    if axes is None:
        axes = tuple(range(image.ndim))

    # Size range:
    sigma_range = (min_sigma, max(min_sigma, max_sigma) + 1e-9)

    # Combine fixed parameters:
    other_fixed_parameters = other_fixed_parameters | {'axes': axes}

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {'sigma': sigma_range}

    # Partial function:
    _denoise_gaussian = partial(denoise_gaussian, **other_fixed_parameters)

    # Calibrate denoiser
    best_parameters = (
        calibrate_denoiser(
            crop,
            _denoise_gaussian,
            mode=optimiser,
            denoise_parameters=parameter_ranges,
            interpolation_mode=jinv_interpolation_mode,
            max_num_evaluations=max_num_evaluations,
            blind_spots=blind_spots,
            display_images=display_images,
            loss_function='L1',
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = 2 * image.nbytes

    return denoise_gaussian, best_parameters, memory_needed


def denoise_gaussian(
    image: ArrayLike,
    axes: Optional[Tuple[int, ...]] = None,
    sigma: float = 1.0,
    truncate: float = 4.0,
    **kwargs,
):
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
    instead unless you have a good reason to use this one.
    \n\n
    Note: We recommend applying a variance stabilisation transform
    to improve results for images with non-Gaussian noise.

    Parameters
    ----------
    image: ArrayLike
        nD image to denoise

    axes: Optional[Tuple[int,...]]
        Axes over which to apply low-pass filtering.
        (advanced)

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

    # Default axes:
    if axes is not None:
        # populate sigma tuple according to axes:
        sigma = tuple((sigma if (i in axes) else 0) for i in range(image.ndim))

    # Gaussian filtering:
    return gaussian_filter(image, sigma=sigma, truncate=truncate)
