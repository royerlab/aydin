from functools import partial
import numpy
from numpy import random

from aydin.it.classic_denoisers.butterworth import denoise_butterworth
from aydin.util.j_invariance.j_invariant_classic import calibrate_denoiser_classic
from aydin.util.crop.rep_crop import representative_crop


def resolution_estimate(image, precision: float = 0.01, display_images: bool = False):
    """Estimation of isotropic resolution in normalised frequency (within [0, 1]).
    Work best

    Parameters
    ----------
    image : numpy.typing.ArrayLike
        Image to estimate (isotropic) resolution from
    precision: float
        Precision of the search
    display_images: bool
        Display image, for debugging purposes.

    Returns
    -------
    Estimate of resolution in normalised frequency -- which assumes that the
    sampling frequency is 1. Values are between 0 and 1, with 0 meaning no
    resolution at all, and 1 means full resolution.
    Also returns the crop used for the estimate.

    """

    # obtain representative crop, to speed things up...
    crop = representative_crop(image, crop_size=128000, equal_sides=True)

    # Convert image to float if needed:
    crop = crop.astype(dtype=numpy.float32, copy=False)

    # We need to add noise for this to work:
    sigma = 0.01 * numpy.std(crop)
    crop += random.normal(scale=sigma, size=crop.shape)

    # ranges:
    freq_cutoff_range = numpy.arange(0.001, 1.0, precision)

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {'freq_cutoff': freq_cutoff_range}

    # Partial function:
    _denoise_sobolev = partial(denoise_butterworth, multi_core=False, order=2)

    # Calibrate denoiser
    best_parameters = calibrate_denoiser_classic(
        crop,
        _denoise_sobolev,
        denoise_parameters=parameter_ranges,
        display_images=display_images,
    )

    norm_frequency = best_parameters.pop('freq_cutoff')

    return norm_frequency, crop
