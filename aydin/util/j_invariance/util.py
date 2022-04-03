import itertools
import traceback

import numpy
from numpy import zeros_like
from numpy.typing import ArrayLike
from scipy.ndimage import convolve, gaussian_filter, median_filter

from aydin.analysis.blind_spot_analysis import auto_detect_blindspots
from aydin.util.log.log import lprint


def _j_invariant_loss(
    image, interp, denoise_function, mask, loss_function, denoiser_kwargs=None
):
    image = image.astype(dtype=numpy.float32, copy=False)

    try:
        denoised = _invariant_denoise(
            image,
            interp,
            denoise_function=denoise_function,
            mask=mask,
            denoiser_kwargs=denoiser_kwargs,
        )
    except RuntimeError:
        lprint(
            "Denoising failed during calibration, skipping by returning blank image "
        )
        print(traceback.format_exc())
        denoised = numpy.zeros_like(image)

    loss = loss_function(image[mask], denoised[mask])

    return loss


def _invariant_denoise(image, interp, denoise_function, mask, *, denoiser_kwargs=None):

    image = image.astype(dtype=numpy.float32, copy=False)

    if denoiser_kwargs is None:
        denoiser_kwargs = {}

    input_image = image.copy()
    input_image[mask] = interp[mask]
    output = denoise_function(input_image, **denoiser_kwargs)

    return output


def _interpolate_image(image: ArrayLike, mode: str):

    if mode == 'gaussian':
        conv_filter = numpy.zeros(shape=(3,) * image.ndim, dtype=image.dtype)
        conv_filter.ravel()[conv_filter.size // 2] = 1
        conv_filter = gaussian_filter(conv_filter, sigma=0.5)
        conv_filter.ravel()[conv_filter.size // 2] = 0
        conv_filter /= conv_filter.sum()

        interpolation = convolve(image, conv_filter, mode='mirror')

    elif mode == 'median':

        footprint = numpy.ones(shape=(3,) * image.ndim, dtype=image.dtype)
        footprint.ravel()[footprint.size // 2] = 0

        interpolation = median_filter(image, footprint=footprint, mode='mirror')

    return interpolation


def _generate_mask(
    image: ArrayLike,
    stride: int = 4,
    max_range: int = 4,
    enable_extended_blind_spot: bool = True,
):

    # Generate slice for mask:
    spatialdims = image.ndim
    n_masks = stride ** spatialdims
    mask = _generate_grid_slice(
        image.shape[:spatialdims], offset=n_masks // 2, stride=stride
    )

    # Do we have to extend these spots?
    if enable_extended_blind_spot:
        lprint(f"Detection of extended blindspots requested!")
        blind_spots, noise_auto = auto_detect_blindspots(image, max_range=max_range)
        extended_blind_spot = len(blind_spots) > 1 and enable_extended_blind_spot
    else:
        extended_blind_spot = False

    if extended_blind_spot:
        lprint(f"Extended blindspots detected: {blind_spots}")
        # If yes, we need to change the way we store the mask:
        mask_full = zeros_like(image, dtype=numpy.bool_)
        mask_full[mask] = True
        mask = mask_full

        spot_kernel = zeros_like(noise_auto, dtype=numpy.bool_)
        for blind_spot in blind_spots:
            blind_spot = tuple(
                slice(s // 2 + x, s // 2 + x + 1)
                for s, x in zip(spot_kernel.shape, blind_spot)
            )
            spot_kernel[blind_spot] = True

        # We extend the spots:
        mask = convolve(mask, spot_kernel)

    return mask


def _generate_grid_slice(shape, *, offset, stride=3):
    phases = numpy.unravel_index(offset, (stride,) * len(shape))
    mask = tuple(slice(p, None, stride) for p in phases)
    return mask


def _mid_point(numerical_parameters_bounds):
    mid_point = tuple(0.5 * (u + v) for u, v in numerical_parameters_bounds)
    mid_point = numpy.array(mid_point)
    return mid_point


def _product_from_dict(dictionary):
    """Utility function to convert parameter ranges to parameter combinations.

    Converts a dict of lists into a list of dicts whose values consist of the
    cartesian product of the values in the original dict.

    Parameters
    ----------
    dictionary : dict of lists
        Dictionary of lists to be multiplied.

    Yields
    ------
    selections : dicts of values
        Dicts containing individual combinations of the values in the input
        dict.
    """
    keys = dictionary.keys()
    for element in itertools.product(*dictionary.values()):
        yield dict(zip(keys, element))
