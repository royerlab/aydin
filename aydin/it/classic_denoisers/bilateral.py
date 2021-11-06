from functools import partial
from typing import Optional
import numpy
import numpy as np
from skimage.restoration import denoise_bilateral as skimage_denoise_bilateral

from aydin.util.crop.rep_crop import representative_crop
from aydin.util.denoise_nd.denoise_nd import extend_nd
from aydin.util.j_invariance.j_invariant_classic import calibrate_denoiser_classic


def calibrate_denoise_bilateral(
    image,
    bins: int = 10000,
    crop_size_in_voxels: Optional[int] = None,
    display_images: bool = False,
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

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate
        denoiser.

    display_images: bool
        When True the denoised images encountered during
        optimisation are shown

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

    # Sigma spatial range:
    sigma_spatial_range = np.arange(0.01, 1, 0.05) ** 1.5

    # Sigma color range:
    sigma_color_range = np.arange(0.01, 1, 0.05) ** 1.5

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {
        'sigma_spatial': sigma_spatial_range,
        'sigma_color': sigma_color_range,
    }

    # Combine fixed parameters:
    other_fixed_parameters = other_fixed_parameters | {'bins': bins}

    # Partial function:
    _denoise_bilateral = partial(denoise_bilateral, **other_fixed_parameters)

    # Calibrate denoiser
    best_parameters = (
        calibrate_denoiser_classic(
            crop,
            _denoise_bilateral,
            denoise_parameters=parameter_ranges,
            display_images=display_images,
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = 2 * image.nbytes

    return denoise_bilateral, best_parameters, memory_needed


def denoise_bilateral(image, sigma_color=None, sigma_spatial=1, bins=10000, **kwargs):
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
