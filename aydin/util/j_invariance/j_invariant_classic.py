import itertools
import math
from functools import partial

import numpy
import numpy as np
from numpy import zeros_like
from scipy import ndimage as ndi
from scipy.ndimage import convolve
from skimage.metrics import mean_squared_error

from aydin.analysis.blind_spot_analysis import auto_detect_blindspots
from aydin.util.log.log import lsection, lprint


def calibrate_denoiser_classic(
    image,
    denoise_function,
    denoise_parameters,
    *,
    stride=4,
    loss_function=mean_squared_error,
    # _structural_loss, mean_squared_error, mean_absolute_error #
    display_images=False,
    **other_fixed_parameters,
):
    """
    Calibrates denoiser using self-supervised loss from Batson & Royer*
    Derived from code here:
    https://scikit-image.org/docs/dev/auto_examples/filters/plot_j_invariant_tutorial.html
    Reference: "Noise2Self: Blind Denoising by Self-Supervision, International
    Conference on Machine Learning, p. 524-533 (2019)"

    This 'classic_denoisers' version uses a 'brute-force' optimizer. Good when the
    denoiser is fast enough and the parameter space to explore small enough.

    Parameters
    ----------
    image: ArrayLike
        Image to calibate denoiser with.
    denoise_function: Callable
        Denosing function to calibrate. Should take an image as first parameter,
        all other parameters should have defaults
    denoise_parameters:
        Dictionary with keys corresponding to parameters of the denoising function.
        Values are either: (i) a list of possible values (categorical parameter),
        or (ii) a tuple of floats defining the bounds of that numerical parameter.
    stride: int
        Stride to compute self-supervised loss.
    loss_function: Callable
        Loss/Error function: takes two arrays and returns a distance-like function.
        Can be:  structural_error, mean_squared_error, _mean_absolute_error
    display_images: bool
        If True the denoised images for each parameter tested are displayed.
        this _will_ be slow.
    other_fixed_parameters: dict
        Other fixed parameters to pass to the denoiser function.


    Returns
    -------
    Dictionary with optimal parameters

    """

    lprint(f"Calibrating denoiser on image of shape: {image.shape}")
    lprint(f"Stride for Noise2Self loss: {stride}")
    lprint(f"Fixed parameters: {other_fixed_parameters}")

    # Pass fixed parameters:
    denoise_function = partial(denoise_function, **other_fixed_parameters)

    parameters_tested, losses = _calibrate_denoiser_search(
        image,
        denoise_function,
        denoise_parameters=denoise_parameters,
        stride=stride,
        loss_function=loss_function,
        display_images=display_images,
    )

    idx = np.argmin(losses)
    best_parameters = parameters_tested[idx]

    lprint(f"Best parameters are: {best_parameters}")

    return best_parameters | other_fixed_parameters


def _j_invariant_loss(
    image,
    denoise_function,
    mask,
    loss_function=mean_squared_error,  # _structural_loss, #
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

    interp = _interpolate_image(image, mask=mask)
    output = np.zeros_like(image)

    input_image = image.copy()
    input_image[mask] = interp[mask]
    output[mask] = denoise_function(input_image, **denoiser_kwargs)[mask]

    return output


def _interpolate_image(image, mask, num_iterations: int = 1):

    # if isinstance(mask, numpy.ndarray):
    #     interpolation = uniform_filter_auto(image, size=3)
    #     if num_iterations > 1:
    #         for _ in range(num_iterations-1):
    #             interpolation[~mask] = image
    #             interpolation = uniform_filter_auto(interpolation)
    # else:
    #     interpolation = uniform_filter_auto(image, size=3)

    conv_filter = ndi.generate_binary_structure(image.ndim, 1).astype(image.dtype)
    conv_filter.ravel()[conv_filter.size // 2] = 0
    conv_filter /= conv_filter.sum()

    interpolation = ndi.convolve(image, conv_filter, mode='mirror')

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
    phases = np.unravel_index(offset, (stride,) * len(shape))
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


def _calibrate_denoiser_search(
    image,
    denoise_function,
    denoise_parameters,
    *,
    stride=4,
    loss_function=mean_squared_error,  # _structural_loss, #
    display_images: bool = False,
):
    """Return a parameter search history with losses for a denoise function.

    Parameters
    ----------
    image : ndarray
        Input data to be denoised (converted using `img_as_float`).
    denoise_function : function
        Denoising function to be calibrated.
    denoise_parameters : dict of list
        Ranges of parameters for `denoise_function` to be calibrated over.
    stride : int, optional
        Stride used in masking procedure that converts `denoise_function`
        to J-invariance.


    Returns
    -------
    parameters_tested : list of dict
        List of parameters tested for `denoise_function`, as a dictionary of
        kwargs.
    losses : list of int
        Self-supervised loss for each set of parameters in `parameters_tested`.
    """
    # image = img_as_float(image)
    parameters_tested = list(_product_from_dict(denoise_parameters))
    losses = []
    denoised_images = []

    with lsection(
        f"Searching for best denoising parameters among: {denoise_parameters}"
    ):

        # Generate mask:
        mask = _generate_mask(image, stride)

        for denoiser_kwargs in parameters_tested:
            with lsection(f"computing J-inv loss for: {denoiser_kwargs}"):

                # denoised = _invariant_denoise(
                #     image, denoise_function, mask, denoiser_kwargs=denoiser_kwargs
                # )
                # loss = loss_function(image[mask], denoised[mask])

                # We compute the J-inv loss:
                loss = _j_invariant_loss(
                    image,
                    denoise_function,
                    mask=mask,
                    loss_function=loss_function,
                    denoiser_kwargs=denoiser_kwargs,
                )

                if math.isnan(loss) or math.isinf(loss):
                    loss = math.inf
                lprint(f"J-inv loss is: {loss}")
                losses.append(loss)
                if display_images and not (math.isnan(loss) or math.isinf(loss)):
                    denoised = denoise_function(image, **denoiser_kwargs)
                    denoised_images.append(denoised)

    if display_images:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name='image')
            viewer.add_image(numpy.stack(denoised_images), name='denoised')

    return parameters_tested, losses
