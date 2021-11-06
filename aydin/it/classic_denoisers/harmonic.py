import math
from functools import partial
from typing import Optional

import numpy
from numpy.linalg import norm
from scipy.ndimage import (
    median_filter,
    minimum_filter,
    maximum_filter,
    uniform_filter,
    rank_filter,
    gaussian_filter,
)

from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariant_smart import calibrate_denoiser_smart
from aydin.util.log.log import lprint


def calibrate_denoise_harmonic(
    image,
    rank: bool = False,
    crop_size_in_voxels: Optional[int] = None,
    max_num_evaluations: int = 16,
    display_images: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates the Harmonic denoiser for the given image and returns the optimal
    parameters obtained using the N2S loss.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate denoiser for.

    rank: bool
        If true, uses a better estimate of min and max filters using a rank
        filter, may be much slower.
        (advanced)

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate denoiser.
        (advanced)

    max_num_evaluations: int
        Maximum number of evaluations for finding the optimal parameters.
        (advanced)

    display_images: bool
        When True the denoised images encountered during optimisation are shown

    other_fixed_parameters: dict
        Any other fixed parameters

    Returns
    -------
    Denoising function, dictionary containing optimal parameters,
    and free memory needed in bytes for computation.
    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # obtain representative crop, to speed things up...
    crop = representative_crop(image, crop_size=crop_size_in_voxels)

    # alpha range:
    # alpha_range = numpy.concatenate(
    #     [numpy.arange(0.01, 1, 0.1), 0.5 + 0.15 * numpy.arange(-1, 1, 0.1)]
    # )
    alpha_range = (0.0, 1.0)

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {
        'alpha': alpha_range,
        'filter': ['uniform', 'gaussian', 'median'],
    }

    # Combine fixed parameters:
    other_fixed_parameters = other_fixed_parameters | {'rank': rank}

    # Partial function:
    _denoise_harmonic = partial(denoise_harmonic, **other_fixed_parameters)

    # Calibrate denoiser
    best_parameters = (
        calibrate_denoiser_smart(
            crop,
            _denoise_harmonic,
            denoise_parameters=parameter_ranges,
            max_num_evaluations=max_num_evaluations,
            display_images=display_images,
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = 3 * image.nbytes

    return denoise_harmonic, best_parameters, memory_needed


def denoise_harmonic(
    image,
    filter: str = 'uniform',
    rank: bool = False,
    max_iterations: int = 1024,
    epsilon: float = 1e-8,
    alpha: float = 0.5,
    step: float = 0.1,
    max_gradient: float = 32,
    **kwargs,
):
    """
    Denoises the given image by applying a non-linear <a
    href="https://en.wikipedia.org/wiki/Harmonic_function">harmonic</a>
    prior.
    The gradient of the prior term is a laplace-like operator
    that is scaled with the range of values around the pixel. One
    of its specificities is that it is very effective at smoothing
    uniform regions of an image.
    \n\n
    Note: Works well in practice, but a bit slow as currently
    implemented for large images.

    Parameters
    ----------
    image: ArrayLike
        Image to be denoised

    filter: str
        Type of filter used to compute the Laplace-like prior. Can be
        either 'uniform', 'gaussian' or 'median'.

    rank: bool
        Uses a more robust estimation of the range.

    max_iterations: int
        Max number of  iterations.

    epsilon: float
        Small value used to determine convergence.

    alpha: float
        Balancing between data and prior term. A value of 0 favours the prior
        entirely, whether a value of 1 favours the data term entirely.

    step: float
        Starting step used for gradient descent

    max_gradient: float
        Maximum gradient tolerated -- usefull to avoid 'numerical explosions' ;-)

    Returns
    -------
    Denoised image
    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # Allocate denoise image:
    denoised = numpy.empty_like(image)
    denoised[...] = image

    # allocate gradient:
    gradient = numpy.zeros_like(image)

    # last gradient magnitude:
    last_gradient_norm = math.inf

    # Iterations:
    counter = 0
    for i in range(max_iterations):

        # regularisation term:
        gradient[...] = alpha * _laplace(denoised, filter, rank)

        # data term:
        gradient += (1 - alpha) * (image - denoised)

        # clip gradient:
        gradient.clip(-max_gradient, max_gradient)

        # Optimisation step:
        denoised += step * gradient

        # gradient magnitude:
        gradient_norm = norm(gradient.ravel(), ord=numpy.inf) / gradient.size

        if gradient_norm > last_gradient_norm:
            # If gradient jumps around turn down the heat:
            step *= 0.95
        else:
            # if teh gradient goes down, turn up the heat:
            step *= 1.01

        # Stopping if convergence or stabilisation:
        if gradient_norm < epsilon:
            lprint("Converged!")
            break

        # Stopping if convergence or stabilisation:
        if last_gradient_norm == gradient_norm:
            counter += 1
            if counter > 8:
                lprint("Gradient stabilised! converged!")
                break

        # We save the last gradient:
        last_gradient_norm = gradient_norm

        # if i % 32 == 0:
        #    lprint(f"gradient magnitude: {gradient_norm}, step: {step}")

    return denoised


def _laplace(image, filter: str = 'uniform', rank: bool = False):
    if rank:
        min_value = rank_filter(image, size=3, rank=1)
        max_value = rank_filter(image, size=3, rank=-2)
    else:
        min_value = minimum_filter(image, size=3)
        max_value = maximum_filter(image, size=3)

    range = max_value - min_value + 1e-6

    if filter == 'uniform':
        return (uniform_filter(image, size=3) - image) / range
    elif filter == 'gaussian':
        return (gaussian_filter(image, sigma=1, truncate=2) - image) / range
    elif filter == 'median':
        return (median_filter(image, size=3) - image) / range
