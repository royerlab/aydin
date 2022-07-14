import math
from functools import partial
from typing import Optional, List, Tuple

import numpy
from numpy.linalg import norm
from numpy.typing import ArrayLike
from scipy.ndimage import (
    median_filter,
    minimum_filter,
    maximum_filter,
    uniform_filter,
    rank_filter,
    gaussian_filter,
)

from aydin.it.classic_denoisers import _defaults
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariance import calibrate_denoiser
from aydin.util.log.log import lprint


def calibrate_denoise_harmonic(
    image: ArrayLike,
    rank: bool = False,
    crop_size_in_voxels: Optional[int] = _defaults.default_crop_size_verylarge.value,
    optimiser: str = _defaults.default_optimiser.value,
    max_num_evaluations: int = _defaults.default_max_evals_hyperlow.value,
    blind_spots: Optional[List[Tuple[int]]] = _defaults.default_blind_spots.value,
    jinv_interpolation_mode: str = _defaults.default_jinv_interpolation_mode.value,
    display_images: bool = False,
    display_crop: bool = False,
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
        Increase this number by factors of two if denoising quality is
        unsatisfactory -- this can be important for very noisy images.
        Values to try are: 65000, 128000, 256000, 320000.
        We do not recommend values higher than 512000.

    optimiser: str
        Optimiser to use for finding the best denoising
        parameters. Can be: 'smart' (default), or 'fast' for a mix of SHGO
        followed by L-BFGS-B.
        (advanced)

    max_num_evaluations: int
        Maximum number of evaluations for finding the optimal parameters.
        Increase this number by factors of two if denoising quality is
        unsatisfactory.

    blind_spots: bool
        List of voxel coordinates (relative to receptive field center) to
        be included in the blind-spot. For example, you can give a list of
        3 tuples: [(0,0,0), (0,1,0), (0,-1,0)] to extend the blind spot
        to cover voxels of relative coordinates: (0,0,0),(0,1,0), and (0,-1,0)
        (advanced) (hidden)

    jinv_interpolation_mode: str
        J-invariance interpolation mode for masking. Can be: 'median' or
        'gaussian'.
        (advanced)

    display_images: bool
        When True the denoised images encountered during optimisation are shown.
        (advanced) (hidden)

    display_crop: bool
        Displays crop, for debugging purposes...
        (advanced) (hidden)

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
    crop = representative_crop(
        image, crop_size=crop_size_in_voxels, display_crop=display_crop
    )

    # Combine fixed parameters:
    other_fixed_parameters = other_fixed_parameters | {'rank': rank}

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {
        'alpha': numpy.linspace(0, 1, max_num_evaluations).tolist(),
        'filter': ['uniform'],
    }

    # Partial function:
    _denoise_harmonic = partial(denoise_harmonic, **other_fixed_parameters)

    # Calibrate denoiser 1st pass:
    best_parameters = (
        calibrate_denoiser(
            crop,
            _denoise_harmonic,
            mode=optimiser,
            denoise_parameters=parameter_ranges,
            interpolation_mode=jinv_interpolation_mode,
            max_num_evaluations=max_num_evaluations,
            blind_spots=blind_spots,
            display_images=display_images,
            loss_function='L2',
        )
        | other_fixed_parameters
    )

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {
        'alpha': [best_parameters['alpha']],
        'filter': ['uniform', 'gaussian', 'median'],
    }

    # Calibrate denoiser 2nd pass:
    best_parameters = (
        calibrate_denoiser(
            crop,
            _denoise_harmonic,
            mode=optimiser,
            denoise_parameters=parameter_ranges,
            interpolation_mode=jinv_interpolation_mode,
            max_num_evaluations=max_num_evaluations,
            blind_spots=blind_spots,
            display_images=display_images,
            loss_function='L2',
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = 3 * image.nbytes

    return denoise_harmonic, best_parameters, memory_needed


def denoise_harmonic(
    image: ArrayLike,
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
