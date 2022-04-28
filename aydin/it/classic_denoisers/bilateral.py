from functools import partial
from typing import Optional, List, Tuple

import numpy
from numpy.typing import ArrayLike
from skimage.restoration import denoise_bilateral as skimage_denoise_bilateral

from aydin.it.classic_denoisers import _defaults
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.denoise_nd.denoise_nd import extend_nd
from aydin.util.j_invariance.j_invariance import calibrate_denoiser


def calibrate_denoise_bilateral(
    image: ArrayLike,
    bins: int = 10000,
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
    Calibrates the bilateral denoiser for the given image and returns the optimal
    parameters obtained using the N2S loss.

    Note: it seems that the bilateral filter of scikit-image
    is broken!

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate denoiser for.

    bins: int
        Number of discrete values for Gaussian weights of
        color filtering. A larger value results in improved
        accuracy.
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
        When True the denoised images encountered during
        optimisation are shown
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

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {'sigma_spatial': (0.01, 1), 'sigma_color': (0.01, 1)}

    # Combine fixed parameters:
    other_fixed_parameters = other_fixed_parameters | {'bins': bins}

    # Partial function:
    _denoise_bilateral = partial(denoise_bilateral, **other_fixed_parameters)

    # Calibrate denoiser
    best_parameters = (
        calibrate_denoiser(
            crop,
            _denoise_bilateral,
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
    memory_needed = 2 * image.nbytes

    return denoise_bilateral, best_parameters, memory_needed


def denoise_bilateral(
    image: ArrayLike,
    sigma_color: Optional[float] = None,
    sigma_spatial: float = 1,
    bins: int = 10000,
    **kwargs,
):
    """
    Denoises the given image using a <a
    href="https://en.wikipedia.org/wiki/Bilateral_filter">bilateral
    filter</a>.
    The bilateral filter is a edge-preserving smoothing filter that can
    be used for image denoising. Each pixel value is replaced by a
    weighted average of intensity values from nearby pixels. The
    weighting is inversely related to the pixel distance in space but
    also in the pixels value differences.

    Parameters
    ----------
    image : ArrayLike
        Image to denoise

    sigma_color : float
        Standard deviation for grayvalue/color distance (radiometric
        similarity). A larger value results in averaging of pixels with larger
        radiometric differences. Note, that the image will be converted using
        the `img_as_float` function and thus the standard deviation is in
        respect to the range ``[0, 1]``. If the value is ``None`` the standard
        deviation of the ``image`` will be used.

    sigma_spatial : float
        Standard deviation for range distance. A larger value results in
        averaging of pixels with larger spatial differences.

    bins : int
        Number of discrete values for Gaussian weights of color filtering.
        A larger value results in improved accuracy.

    kwargs: dict
        Other parameters

    Returns
    -------
    Denoised image

    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    _skimage_denoise_bilateral = extend_nd(available_dims=[2])(
        skimage_denoise_bilateral
    )

    return _skimage_denoise_bilateral(
        image,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        bins=bins,
        mode='reflect',
        **kwargs,
    )
