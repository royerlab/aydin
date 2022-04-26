import itertools
import traceback
from functools import lru_cache, reduce
from typing import Optional, List, Tuple

import numpy
from numpy import zeros_like, zeros
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
        if denoiser_kwargs is None:
            denoiser_kwargs = {}

        denoised = denoise_function(masked_input_image, **denoiser_kwargs)
    except RuntimeError:
        lprint(
            "Denoising failed during calibration, skipping by returning "
            "the original image (not denoised)."
        )
        print(traceback.format_exc())
        denoised = image.copy()

    loss = loss_function(image[mask], denoised[mask])

    return loss


def _interpolate_image(
    image: ArrayLike,
    mask: ArrayLike,
    mode: str,
    num_iterations: int,
    boundary_mode: str = 'mirror',
):

    # Prepare kernels and footprints:
    gaussian_kernel = _generate_gaussian_kernel(image.ndim, image.dtype)
    median_footprint = _generate_median_footprint(image.ndim, image.dtype)

    # Prepare initial interpolation:
    interpolation = numpy.zeros_like(image)
    interpolation[...] = image

    # Ensure no J-invariant leakage:
    interpolation[mask] = 0

    # Run multiple iterations of the interpolation by diffusion:
    for i in range(num_iterations):

        # Aply diffusion step:
        if mode == 'gaussian':
            interpolation = convolve(interpolation, gaussian_kernel, mode=boundary_mode)
        elif mode == 'median':
            interpolation = median_filter(
                interpolation, footprint=median_footprint, mode=boundary_mode
            )

        # Enforce that non-masked values are set to the original's image values:
        interpolation[~mask] = image[~mask]

    return interpolation


@lru_cache(maxsize=None)
def _generate_gaussian_kernel(ndim: int, dtype, size: int = 5, sigma: float = 0.75):

    kernel = numpy.zeros(shape=(size,) * ndim, dtype=dtype)
    kernel.ravel()[kernel.size // 2] = 1
    kernel = gaussian_filter(kernel, sigma=sigma)
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
    blind_spots: Optional[List[Tuple[int]]] = None,
):

    # Generate slice for mask:
    spatialdims = image.ndim

    # Compute slicing for mask:
    n_masks = stride ** spatialdims
    mask_slice = _generate_grid_slice(
        image.shape[:spatialdims], offset=n_masks // 2, stride=stride
    )

    # Compute default mask:
    mask = zeros_like(image, dtype=numpy.bool_)
    mask[mask_slice] = True

    # Do we have to extend these spots?
    if blind_spots is None:
        lprint("Detection of extended blindspots requested!")
        blind_spots, noise_auto = auto_detect_blindspots(image, max_range=max_range)

    extended_blind_spot = len(blind_spots) > 1

    if extended_blind_spot:
        lprint(f"Extended blindspots detected: {blind_spots}")

        # Determine size of spot kernel:
        min_coord = tuple(
            reduce(lambda u, v: (min(a, b) for a, b in zip(u, v)), blind_spots)
        )
        max_coord = tuple(
            reduce(lambda u, v: (max(a, b) for a, b in zip(u, v)), blind_spots)
        )
        spot_kernel_shape = tuple(
            (1 + 2 * max(abs(u), abs(v)) for u, v in zip(max_coord, min_coord))
        )

        spot_kernel = zeros(spot_kernel_shape, dtype=numpy.bool_)
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
