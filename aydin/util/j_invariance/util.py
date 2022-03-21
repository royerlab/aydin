import itertools

import numpy
from numpy import zeros_like
from scipy.ndimage import generate_binary_structure, convolve

from aydin.analysis.blind_spot_analysis import auto_detect_blindspots


def _j_invariant_loss(
    image,
    denoise_function,
    mask,
    loss_function,
    denoiser_kwargs=None,
):
    image = image.astype(dtype=numpy.float32, copy=False)

    denoised = _invariant_denoise(
        image,
        denoise_function=denoise_function,
        mask=mask,
        denoiser_kwargs=denoiser_kwargs,
    )

    loss = loss_function(image[mask], denoised[mask])

    return loss


def _invariant_denoise(image, denoise_function, mask, *, denoiser_kwargs=None):

    image = image.astype(dtype=numpy.float32, copy=False)

    if denoiser_kwargs is None:
        denoiser_kwargs = {}

    interp = _interpolate_image(image)
    output = numpy.zeros_like(image)

    input_image = image.copy()
    input_image[mask] = interp[mask]
    output[mask] = denoise_function(input_image, **denoiser_kwargs)[mask]

    return output


def _interpolate_image(image):

    conv_filter = generate_binary_structure(image.ndim, 1).astype(image.dtype)
    conv_filter.ravel()[conv_filter.size // 2] = 0
    conv_filter /= conv_filter.sum()

    interpolation = convolve(image, conv_filter, mode='mirror')

    return interpolation


def _generate_mask(image, stride: int = 4, max_range: int = 4):

    # Generate slice for mask:
    spatialdims = image.ndim
    n_masks = stride ** spatialdims
    mask = _generate_grid_slice(
        image.shape[:spatialdims], offset=n_masks // 2, stride=stride
    )

    # Do we have to extend these spots?
    blind_spots, noise_auto = auto_detect_blindspots(image, max_range=max_range)
    extended_blind_spot = len(blind_spots) > 1

    if extended_blind_spot:
        # If yes, we need to chnage the way we store the mask:
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
