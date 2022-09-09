from functools import partial

import numpy
from numpy import random

from aydin.it.classic_denoisers.butterworth import denoise_butterworth
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariance import calibrate_denoiser


def resolution_estimate(image, precision: int = 2, display_images: bool = False):
    """Estimation of isotropic resolution in normalised frequency (within [0, 1]).
    Work best

    Parameters
    ----------
    image : numpy.typing.ArrayLike
        Image to estimate (isotropic) resolution from
    precision: int
        Precision in decimal digits
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
    crop_original = representative_crop(image, crop_size=256000, equal_sides=True)

    # Convert image to float if needed:
    crop_original = crop_original.astype(dtype=numpy.float32, copy=False)

    # We need to add a bit of noise for this to work:
    sigma = 1e-9 * numpy.std(crop_original)
    crop = crop_original + random.normal(scale=sigma, size=crop_original.shape)

    step = 0.1
    frequency = 0.5

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

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(image, name='image')
        # viewer.add_image(crop_original, name='crop_original')
        # viewer.add_image(crop, name='crop')
        # napari.run()

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

    return frequency, crop_original
