import math
from functools import partial
from typing import List

import numpy
import numpy as np
from numpy.typing import ArrayLike

from aydin.util.j_invariance.losses import mean_squared_error, mean_absolute_error
from aydin.util.j_invariance.util import (
    _product_from_dict,
    _generate_mask,
    _j_invariant_loss,
)
from aydin.util.log.log import lsection, lprint


def calibrate_denoiser_classic(
    image,
    denoise_function,
    denoise_parameters,
    stride=4,
    loss_function: str = 'L2',
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
    loss_function: str
        Loss/Error function: Can be:  'L1', 'L2', 'SSIM'
    display_images: bool
        If True the denoised images for each parameter tested are displayed.
        this _will_ be slow.
    other_fixed_parameters: dict
        Other fixed parameters to pass to the denoiser function.


    Returns
    -------
    Dictionary with optimal parameters

    """

    # Loss function:
    if loss_function == 'L1':
        loss_function = mean_absolute_error
    elif loss_function == 'L2':
        loss_function = mean_squared_error
    elif loss_function == 'SSIM':
        loss_function = mean_squared_error
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")

    lprint(f"Calibrating denoiser on image of shape: {image.shape}")
    lprint(f"Stride for Noise2Self loss: {stride}")
    lprint(f"Fixed parameters: {other_fixed_parameters}")

    # Pass fixed parameters:
    denoise_function = partial(denoise_function, **other_fixed_parameters)

    # image = img_as_float(image)
    parameters_tested = list(_product_from_dict(denoise_parameters))
    losses = []
    denoised_images: List[ArrayLike] = []

    # Generate mask:
    mask = _generate_mask(image, stride)

    with lsection(
        f"Searching for best denoising parameters among: {denoise_parameters}"
    ):

        for denoiser_kwargs in parameters_tested:
            with lsection(f"computing J-inv loss for: {denoiser_kwargs}"):

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

    idx = np.argmin(losses)
    best_parameters = parameters_tested[idx]

    lprint(f"Best parameters are: {best_parameters}")

    if display_images:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name='image')
            viewer.add_image(numpy.stack(denoised_images), name='denoised')

    return best_parameters | other_fixed_parameters
