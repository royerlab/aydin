from functools import partial
from typing import Optional

import numpy
import numpy as np
from skimage.restoration import denoise_wavelet as skimage_denoise_wavelet
from skimage.restoration import estimate_sigma

from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariant_classic import calibrate_denoiser_classic


def calibrate_denoise_wavelet(
    image,
    crop_size_in_voxels: Optional[int] = None,
    display_images: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates a <a href="https://en.wikipedia.org/wiki/Wavelet_transform
    ">wavelet</a> denoiser for the given image and returns the optimal
    parameters obtained using the N2S loss.

    Note: we use the scikt-image implementation of wavelet denoising.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate wavelet denoiser for.

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate denoiser.
        (advanced)

    display_images: bool
        When True the denoised images encountered during optimisation are shown

    other_fixed_parameters: dict
        Any other fixed parameters

    Returns
    -------
    Denoising function and dictionary containing optimal parameters.

    """
    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # obtain representative crop, to speed things up...
    crop = representative_crop(image, crop_size=crop_size_in_voxels)

    # We make a first estimate of the noise sigma:
    estimated_sigma = estimate_sigma(image)

    # Sigma range:
    sigma_range = estimated_sigma * np.arange(0.25, 4, 0.01)
    # sigma_range = (0.0, 10.0)

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {
        'sigma': sigma_range,
        'wavelet': [
            'db1',
            'db2',
            'haar',
            'bior4.4',  # same as CDF 9/7 from JPEG 2000 lossy
            # 'sym9',
            # 'coif1',
            # 'coif5',
            # 'dmey',
        ],  # 'bior2.2', 'bior3.1', 'bior3.3',
        'mode': ['soft'],
        'method': ['BayesShrink', 'VisuShrink'],
    }

    # Partial function:
    _denoise_wavelet = partial(denoise_wavelet, **other_fixed_parameters)

    # Calibrate denoiser
    best_parameters = (
        calibrate_denoiser_classic(
            crop,
            _denoise_wavelet,
            denoise_parameters=parameter_ranges,
            display_images=display_images,
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = image.nbytes * 3  # transform

    return denoise_wavelet, best_parameters, memory_needed


def denoise_wavelet(
    image,
    wavelet: str = 'db1',
    sigma: float = None,
    mode: str = 'soft',
    method: str = 'BayesShrink',
    **kwargs,
):
    """
    Denoises the given image using the scikit-image
    implementation of <a href="https://en.wikipedia.org/wiki/Wavelet_transform ">
    wavelet</a> denoising.
    \n\n
    Note: we use the scikt-image implementation of wavelet denoising.

    Parameters
    ----------
    image: ArrayLike
        Image to denoise

    wavelet : string, optional
        The type of wavelet to perform and can be any of the options
        ``pywt.wavelist`` outputs. The default is `'db1'`. For example,
        ``wavelet`` can be any of ``{'db2', 'haar', 'sym9'}`` and many more
          (see PyWavelets documentation).

    sigma : float or list, optional
        The noise standard deviation used when computing the wavelet detail
        coefficient threshold(s). When None (default), the noise standard
        deviation is estimated via the method in [2]_.

    mode : {'soft', 'hard'}, optional
        An optional argument to choose the type of denoising performed. It
        noted that choosing soft thresholding given additive noise finds the
        best approximation of the original image.

    method : {'BayesShrink', 'VisuShrink'}, optional
        Thresholding method to be used. The currently supported methods are
        "BayesShrink" [1]_ and "VisuShrink" [2]_. Defaults to "BayesShrink".

    kwargs : dict
        Any other parameters to be passed to scikit-image implementations

    Returns
    -------
    Denoised image as ndarray

    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    return skimage_denoise_wavelet(
        image, wavelet=wavelet, sigma=sigma, mode=mode, method=method, **kwargs
    )
