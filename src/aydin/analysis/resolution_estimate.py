"""Image resolution estimation using self-supervised calibration.

This module estimates the isotropic resolution of an image by calibrating a
Butterworth denoiser and finding the optimal frequency cutoff. The cutoff
frequency serves as a proxy for the resolution limit of the image.
"""

from functools import partial

import numpy
from numpy import random

from aydin.it.classic_denoisers.butterworth import denoise_butterworth
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariance import calibrate_denoiser
from aydin.util.log.log import aprint, asection


def resolution_estimate(image, precision: int = 2, display_images: bool = False):
    """Estimate the isotropic resolution of an image in normalized frequency.

    Uses self-supervised calibration of a Butterworth denoiser to find the
    optimal frequency cutoff, which serves as a resolution estimate.
    Works best on images with a clear distinction between signal and noise
    in the frequency domain.

    Parameters
    ----------
    image : numpy.typing.ArrayLike
        Image to estimate (isotropic) resolution from.
    precision : int
        Precision in decimal digits. Each additional digit of precision
        requires an additional round of optimization.
    display_images : bool
        If True, displays intermediate images for debugging purposes.

    Returns
    -------
    frequency : float
        Estimated resolution as a normalized frequency in [0, 1], where
        the sampling frequency is 1. A value of 0 means no resolution,
        and 1 means full (Nyquist) resolution.
    crop_original : numpy.ndarray
        The representative crop of the image used for the estimate.
    """

    # obtain representative crop, to speed things up...
    crop_original = representative_crop(image, crop_size=256000, equal_sides=True)

    # Convert image to float if needed:
    crop_original = crop_original.astype(dtype=numpy.float32, copy=False)

    # We need to add a bit of noise for this to work:
    sigma = 1e-9 * numpy.std(crop_original)
    crop = crop_original + random.normal(scale=sigma, size=crop_original.shape)

    step = 0.1
    frequency = 0.5

    with asection("Estimating resolution..."):
        for i in range(precision):

            # Start and stop of range:
            start = max(0.0001, frequency - 5 * step)
            stop = min(1.0, frequency + 5 * step)

            # ranges:
            freq_cutoff_range = list(numpy.arange(start, stop, step))

            # Parameters to test when calibrating the denoising algorithm
            parameter_ranges = {'freq_cutoff': freq_cutoff_range}

            # Partial function:
            _denoise_function = partial(denoise_butterworth, multi_core=True, order=5)

            # Calibrate denoiser
            best_parameters = calibrate_denoiser(
                crop,
                _denoise_function,
                denoise_parameters=parameter_ranges,
                display_images=display_images,
                max_num_evaluations=256,
                patience=64,
                loss_function='L1',
                interpolation_mode='gaussian',
                blind_spots=[],
            )

            frequency = best_parameters.pop('freq_cutoff')
            step *= 0.1

            aprint(
                f"Pass {i + 1}/{precision}, frequency={frequency:.6f}, step={step:.6f}"
            )

        aprint(f"Final resolution estimate: {frequency:.6f}")

    return frequency, crop_original
