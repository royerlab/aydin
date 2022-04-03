import itertools
import traceback
from functools import lru_cache

import numpy
from numpy import zeros_like
from numpy.typing import ArrayLike
from scipy.ndimage import convolve, gaussian_filter, median_filter

from aydin.analysis.blind_spot_analysis import auto_detect_blindspots
from aydin.util.log.log import lprint


def _j_invariant_loss(
    image,
    masked_input_image,
    denoise_function,
    mask,
    loss_function,
    denoiser_kwargs=None,
):
    try:
        denoised = _invariant_denoise(
            masked_input_image=masked_input_image,
            denoise_function=denoise_function,
            denoiser_kwargs=denoiser_kwargs,
        )
    except RuntimeError:
        lprint(
            "Denoising failed during calibration, skipping by returning "
            "the original image (not denoised)."
        )
        print(traceback.format_exc())
        denoised = image.copy()

    loss = loss_function(image[mask], denoised[mask])

    return loss


def _invariant_denoise(masked_input_image, denoise_function, *, denoiser_kwargs=None):

    if denoiser_kwargs is None:
        denoiser_kwargs = {}

    output = denoise_function(masked_input_image, **denoiser_kwargs)

    return output


def _interpolate_image(image: ArrayLike, mode: str, boundary_mode: str = 'mirror'):

    if mode == 'gaussian':
        kernel = _generate_gaussian_kernel(image.ndim, image.dtype)
        interpolation = convolve(image, kernel, mode=boundary_mode)

    elif mode == 'median':
        footprint = _generate_median_footprint(image.ndim, image.dtype)
        interpolation = median_filter(image, footprint=footprint, mode=boundary_mode)

    return interpolation


@lru_cache(maxsize=None)
def _generate_gaussian_kernel(ndim: int, dtype):
    kernel = numpy.zeros(shape=(3,) * ndim, dtype=dtype)
    kernel.ravel()[kernel.size // 2] = 1
    kernel = gaussian_filter(kernel, sigma=0.5)
    kernel.ravel()[kernel.size // 2] = 0
    kernel /= kernel.sum()
    return kernel


@lru_cache(maxsize=None)
def _generate_median_footprint(ndim: int, dtype):
    footprint = numpy.ones(shape=(3,) * ndim, dtype=dtype)
    footprint.ravel()[footprint.size // 2] = 0
    return footprint


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
