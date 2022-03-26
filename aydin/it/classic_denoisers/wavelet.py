from functools import partial
from typing import Optional

import numpy
import pywt
from numpy.typing import ArrayLike
from skimage.restoration import denoise_wavelet as skimage_denoise_wavelet
from skimage.restoration import estimate_sigma

from aydin.it.classic_denoisers import _defaults
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariance import calibrate_denoiser


def calibrate_denoise_wavelet(
    image: ArrayLike,
    all_wavelets: bool = False,
    wavelet_name_filter: str = '',
    crop_size_in_voxels: Optional[int] = _defaults.default_crop_size,
    optimiser: str = _defaults.default_optimiser,
    max_num_evaluations: int = _defaults.default_max_evals_normal,
    enable_extended_blind_spot: bool = True,
    display_images: bool = False,
    display_crop: bool = False,
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

    all_wavelets: bool
        If true then all wavelet transforms are tried during calibration,
        otherwise only a selection that we consider to be the best.
        Note: trying all transforms can take a long time but might find the
        magical transform that will make it work for your data.
        (advanced)

    wavelet_name_filter: str
        Comma separated list of wavelet name substrings. We only keep for
        calibration wavelets which name contains these substrings. Best used
        when using all transforms as starting list to select a family of wavelets,
        or to select a specific one by providing its name in full.
        (advanced)

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate denoiser.
        (advanced)

    optimiser: str
        Optimiser to use for finding the best denoising
        parameters. Can be: 'smart' (default), or 'fast' for a mix of SHGO
        followed by L-BFGS-B.
        (advanced)

    max_num_evaluations: int
        Maximum number of evaluations for finding the optimal parameters.
        (advanced)

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
    Denoising function and dictionary containing optimal parameters.

    """
    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # obtain representative crop, to speed things up...
    crop = representative_crop(
        image, crop_size=crop_size_in_voxels, display_crop=display_crop
    )

    # We make a first estimate of the noise sigma:
    estimated_sigma = estimate_sigma(image)

    # Sigma range:
    sigma_range = (
        max(1e-9, min(0.1, 0.5 * estimated_sigma - 0.1)),
        max(2, 2 * estimated_sigma + 1),
    )

    # Lists of wavelets:
    all_wavelets_list = pywt.wavelist()
    best_wavelets_list = [
        'db1',
        'db2',
        'haar',
        'bior4.4',  # same as CDF 9/7 from JPEG 2000 lossy
        'sym9',
        'coif1',
        'coif5',
        'dmey',
        'bior2.2',
        'bior3.1',
        'bior3.3',
        'bior2.8',
    ]

    # List of wavelets to use:
    wavelet_list = all_wavelets_list if all_wavelets else best_wavelets_list

    # we parse the filter list:
    filters = wavelet_name_filter.split(", ")
    filters = list(f.lower().strip() for f in filters)

    # We only keep wavelets that are :
    wavelet_list = list(w for w in wavelet_list if any(f in w for f in filters))

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {
        'sigma': sigma_range,
        'wavelet': wavelet_list,  #
        'mode': ['soft'],
        'method': [
            'BayesShrink',
        ],
    }

    # Partial function:
    _denoise_wavelet = partial(denoise_wavelet, **other_fixed_parameters)

    # Calibrate denoiser 1st pass:
    best_parameters = (
        calibrate_denoiser(
            crop,
            _denoise_wavelet,
            mode=optimiser,
            denoise_parameters=parameter_ranges,
            max_num_evaluations=max_num_evaluations,
            enable_extended_blind_spot=enable_extended_blind_spot,
            display_images=display_images,
        )
        | other_fixed_parameters
    )

    # Next pass we optimise the mode and method:
    parameter_ranges = {
        'sigma': [
            best_parameters['sigma'],
        ],
        'wavelet': [
            best_parameters['wavelet'],
        ],  #
        'mode': ['soft', 'hard'],
        'method': ['BayesShrink', 'VisuShrink'],
    }

    # Calibrate denoiser 2nd pass:
    best_parameters = (
        calibrate_denoiser(
            crop,
            _denoise_wavelet,
            mode=optimiser,
            denoise_parameters=parameter_ranges,
            max_num_evaluations=max_num_evaluations,
            enable_extended_blind_spot=enable_extended_blind_spot,
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
