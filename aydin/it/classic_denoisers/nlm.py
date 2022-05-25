from functools import partial
from typing import Optional, Tuple, List

import numpy
from numpy.typing import ArrayLike
from skimage.restoration import denoise_nl_means as skimage_denoise_nl_means
from skimage.restoration import estimate_sigma

from aydin.it.classic_denoisers import _defaults
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.denoise_nd.denoise_nd import extend_nd
from aydin.util.j_invariance.j_invariance import calibrate_denoiser


def calibrate_denoise_nlm(
    image: ArrayLike,
    patch_size: int = 7,
    patch_distance: int = 11,
    crop_size_in_voxels: Optional[int] = _defaults.default_crop_size_normal.value,
    optimiser: str = _defaults.default_optimiser.value,
    max_num_evaluations: int = _defaults.default_max_evals_normal.value,
    blind_spots: Optional[List[Tuple[int]]] = _defaults.default_blind_spots.value,
    jinv_interpolation_mode: str = _defaults.default_jinv_interpolation_mode.value,
    display_images: bool = False,
    display_crop: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates the Non-Local Means (NLM) denoiser for the given image and
    returns the optimal parameters obtained using the N2S loss.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate denoiser for.

    patch_size : int, optional
        Size of patches used for denoising.
        (advanced)

    patch_distance : int, optional
        Maximal distance in pixels where to search patches used for denoising.
        (advanced)

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
        Maximum number of evaluations for finding the optimal parameters.
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
        When True the denoised images encountered during optimisation are shown.
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

    # We make a first estimate of the noise sigma:
    estimated_sigma = estimate_sigma(image)

    sigma_range = (max(0.0, 0.2 * estimated_sigma), 4 * estimated_sigma)
    cutoff_distance_range = (max(0.0, 0.2 * estimated_sigma), 4 * estimated_sigma)

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {'sigma': sigma_range, 'cutoff_distance': cutoff_distance_range}

    # Combine fixed parameters:
    other_fixed_parameters = other_fixed_parameters | {
        'patch_size': patch_size,
        'patch_distance': patch_distance,
    }

    # Partial function:
    _denoise_nl_means = partial(denoise_nlm, **other_fixed_parameters)

    # Calibrate denoiser
    best_parameters = (
        calibrate_denoiser(
            crop,
            _denoise_nl_means,
            mode=optimiser,
            denoise_parameters=parameter_ranges,
            interpolation_mode=jinv_interpolation_mode,
            max_num_evaluations=max_num_evaluations,
            blind_spots=blind_spots,
            display_images=display_images,
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = 3 * image.nbytes

    return denoise_nlm, best_parameters, memory_needed


def denoise_nlm(
    image: ArrayLike,
    patch_size: int = 7,
    patch_distance: int = 11,
    cutoff_distance: float = 0.1,
    sigma=0.0,
):
    """
    Denoise given image using either scikit-image implementation
    of <a href="https://en.wikipedia.org/wiki/Non-local_means">Non-Local-Means (NLM)</a>.


    Parameters
    ----------
    image : ArayLike
        Image to be denoised

    patch_size : int, optional
        Size of patches used for denoising.

    patch_distance : int, optional
        Maximal distance in pixels where to search patches used for denoising.

    cutoff_distance : float, optional
        Cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches. A higher h results in a smoother image,
        at the expense of blurring features. For a Gaussian noise of standard
        deviation sigma, a rule of thumb is to choose the value of h to be
        sigma of slightly less.

    sigma : float, optional
        The standard deviation of the (Gaussian) noise.  If provided, a more
        robust computation of patch weights is computed that takes the expected
        noise variance into account (see Notes below).

    Returns
    -------
    Denoised image as ndarray.

    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # Make it work for nD images:
    _skimage_denoise_nl_means = extend_nd(available_dims=[2, 3])(
        skimage_denoise_nl_means
    )

    return _skimage_denoise_nl_means(
        image,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=cutoff_distance,
        sigma=sigma,
    )
