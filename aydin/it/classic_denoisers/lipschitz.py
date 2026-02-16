"""Lipschitz continuity denoiser for impulse/salt-and-pepper noise.

Provides calibration and denoising functions that enforce Lipschitz
continuity on the image. Voxels violating the continuity criterion are
iteratively replaced by a filtered value, effectively removing impulse
noise (salt-and-pepper) while preserving smooth variations.
"""

from typing import List, Optional, Tuple

import numpy
from numba import jit
from numpy.typing import ArrayLike
from scipy.ndimage import median_filter, uniform_filter

from aydin.it.classic_denoisers import _defaults
from aydin.util.log.log import aprint, asection

__fastmath = {'contract', 'afn', 'reassoc'}
__error_model = 'numpy'


def calibrate_denoise_lipschitz(
    image: ArrayLike,
    lipschitz: float = 0.1,
    percentile: float = 0.001,
    alpha: float = 0.1,
    max_num_iterations: int = _defaults.default_max_evals_normal.value,
    blind_spots: Optional[List[Tuple[int]]] = _defaults.default_blind_spots.value,
    **other_fixed_parameters,
):
    """
    Calibrates the Lipschitz denoiser for the given image and returns the
    optimal parameters obtained using the N2S loss.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate Sobolev denoiser for.

    lipschitz : float
        Increase to tolerate more variation, decrease to be
        more aggressive in removing salt & pepper noise.

    percentile : float
        Percentile value used to determine the threshold
        for choosing the worst offending voxels per iteration.
        (advanced)

    alpha : float
        This constant controls the amount of correction per iteration.
        Should be a number within[0, 1]. (advanced)

    max_num_iterations: int
        Maximum number of Lipschitz correction iterations to run. (advanced)

    blind_spots: Optional[List[Tuple[int]]]
        List of voxel coordinates (relative to receptive field center) to
        be included in the blind-spot. For example, you can give a list of
        3 tuples: [(0,0,0), (0,1,0), (0,-1,0)] to extend the blind spot
        to cover voxels of relative coordinates: (0,0,0),(0,1,0), and (0,-1,0)
        (advanced) (hidden)

    other_fixed_parameters: dict
        Any other fixed parameters. (advanced)

    Returns
    -------
    denoise_function : callable
        The ``denoise_lipschitz`` function.
    best_parameters : dict
        Dictionary of optimal denoising parameters.
    memory_needed : int
        Estimated memory needed in bytes for denoising the full image.
    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # Combine fixed parameters:
    other_fixed_parameters = other_fixed_parameters | {
        'lipschitz': lipschitz,
        'percentile': percentile,
        'alpha': alpha,
        'max_num_iterations': max_num_iterations,
    }

    best_parameters = other_fixed_parameters
    aprint(f"Best parameters: {best_parameters}")

    # Memory needed:
    memory_needed = 3 * image.nbytes  # complex numbers and more

    return denoise_lipschitz, best_parameters, memory_needed


def denoise_lipschitz(
    image: ArrayLike,
    lipschitz: float = 0.05,
    percentile: float = 0.01,
    alpha: float = 0.1,
    max_num_iterations: int = 128,
    multi_core: bool = True,
):
    """
    Denoises the given image by correcting voxel values that violate
    <a href="https://en.wikipedia.org/wiki/Lipschitz_continuity">Lipschitz
    continuity</a> criterion. These voxels get
    iteratively replaced by the median. This is repeated for a number
    of iterations (max_num_iterations), until no more changes to
    the image occurs, and making sure that the proportion of
    corrected voxels remains below a given level (max_corrections argument).
    <notgui>

    Parameters
    ----------
    image: ArrayLike
        Image to be denoised

    lipschitz : float
        Increase to tolerate more variation, decrease to be
        more aggressive in removing impulse/salt&pepper noise.

    percentile : float
        Percentile value used to determine the threshold
        for choosing the worst offending voxels per iteration.
        (advanced)

    alpha : float
        This constant controls the amount of correction per iteration.
        Should be a number within[0, 1]. (advanced)

    max_num_iterations: int
        Maximum number of Lipschitz correction iterations to run. (advanced)

    multi_core: bool
        By default we use as many cores as possible, in some cases, for small
        (test) images, it might be faster to run on a single core instead of
        starting the whole parallelization machinery. (advanced)


    Returns
    -------
    numpy.ndarray
        Denoised image as a float32 array.
    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=True)

    # Compute median:
    median_ref = median_filter(image, size=3)

    # Wrap compute_error function:
    wrapped_compute_error = jit(nopython=True, parallel=multi_core)(_compute_error)

    with asection(
        f"Lipschitz denoising (lipschitz={lipschitz}, max_iterations={max_num_iterations})"
    ):
        for i in range(max_num_iterations):

            # Compute median:
            median = uniform_filter(image, size=5)

            # We scale the lipschitz threshold to the image std at '3 sigma' :
            _lipschitz = lipschitz * 6 * image.std()

            # We compute the 'error':
            error = wrapped_compute_error(image, median=median, lipschitz=_lipschitz)

            # We compute the threshold on the basis of the errors,
            # we first tackle the most offending voxels:
            threshold = numpy.percentile(error, q=100 * (1 - percentile))

            # We compute the mask:
            mask = error > threshold

            # count number of corrections for this round:
            num_corrections = numpy.sum(mask)

            # if no corrections made we stop iterating:
            if num_corrections == 0:
                aprint(f"Converged after {i} iterations")
                break

            # We use the median to correct pixels:
            image[mask] = (alpha) * image[mask] + (1 - alpha) * median_ref[mask]

        else:
            aprint(f"Reached maximum iterations ({max_num_iterations})")

    return image


def _compute_error(array, median, lipschitz):
    """Compute the Lipschitz error map between an array and its median.

    Values exceeding the Lipschitz threshold contribute to the error;
    values within the threshold produce zero error.

    Parameters
    ----------
    array : numpy.ndarray
        Input image array.
    median : numpy.ndarray
        Median-filtered version of the array.
    lipschitz : float
        Lipschitz threshold scaled to the image.

    Returns
    -------
    numpy.ndarray
        Error map with zero where Lipschitz continuity is satisfied.
    """
    # we compute the error map:
    error = median - array
    error = numpy.abs(error)
    error = numpy.maximum(error, lipschitz)
    error -= lipschitz
    return error
