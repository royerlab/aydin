from functools import partial
from typing import Optional

import numpy
from skimage.restoration import denoise_nl_means as skimage_denoise_nl_means
from skimage.restoration import estimate_sigma

from aydin.util.crop.rep_crop import representative_crop
from aydin.util.denoise_nd.denoise_nd import extend_nd
from aydin.util.j_invariance.j_invariant_smart import calibrate_denoiser_smart


def calibrate_denoise_nlm(
    image,
    patch_size: int = 7,
    patch_distance: int = 11,
    crop_size_in_voxels: Optional[int] = None,
    max_num_evaluations: int = 512,
    display_images: bool = False,
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
        (advanced)

    max_num_evaluations: int
        Maximum number of evaluations for finding the optimal parameters.
        (advanced)

    display_images: bool
        When True the denoised images encountered during optimisation are shown

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

    # We make a first estimate of the noise sigma:
    estimated_sigma = estimate_sigma(image)

    # Sigma range:
    # sigma_range = estimated_sigma + np.arange(-0.1, 0.1, 0.02)
    # clip sigma range:
    # sigma_range = numpy.clip(sigma_range, 0, 1)

    # h range:
    # h_range = estimated_sigma + np.arange(-0.1, 0.1, 0.02)
    # clip sigma range:
    # h_range = numpy.clip(h_range, 0, 1)

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
        calibrate_denoiser_smart(
            crop,
            _denoise_nl_means,
            denoise_parameters=parameter_ranges,
            display_images=display_images,
            max_num_evaluations=max_num_evaluations,
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = 3 * image.nbytes

    return denoise_nlm, best_parameters, memory_needed


def denoise_nlm(
    image,
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
