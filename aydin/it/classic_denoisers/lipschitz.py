import numpy
from numba import jit
from scipy.ndimage import median_filter, uniform_filter

__fastmath = {'contract', 'afn', 'reassoc'}
__error_model = 'numpy'


def calibrate_denoise_lipschitz(
    image,
    lipschitz: float = 0.1,
    percentile: float = 0.001,
    alpha: float = 0.1,
    max_num_iterations: int = 128,
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


    other_fixed_parameters: dict
        Any other fixed parameters. (advanced)

    Returns
    -------
    Denoising function, dictionary containing optimal parameters,
    and free memory needed in bytes for computation.
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

    # Memory needed:
    memory_needed = 3 * image.nbytes  # complex numbers and more

    return denoise_lipschitz, best_parameters, memory_needed


def denoise_lipschitz(
    image,
    lipschitz: float = 0.05,
    percentile: float = 0.01,
    alpha: float = 0.1,
    max_num_iterations: int = 128,
    multi_core: bool = True,
):
    """
    Denoises the given image by correcting voxel values that violate
    <a href="https://en.wikipedia.org/wiki/Lipschitz_continuity">Lipschitz
    continuity</a> criterion. These voxels get replaced
    iteratively replaced by the median. This is repeated for a number
    of iterations (max_num_iterations), until no more changes to
    the image occurs, and making sure that the proportion of
    corrected voxels remains below a given level (max_corrections argument).


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
        This constant controls the amount of correction per ietartion.
        Should be a number within[0, 1]. (advanced)

    max_num_iterations: int
        Maximum number of Lipschitz correction iterations to run. (advanced)

    multi_core: bool
        By default we use as many cores as possible, in some cases, for small
        (test) images, it might be faster to run on a single core instead of
        starting the whole parallelization machinery. (advanced)


    Returns
    -------
    Denoised image

    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=True)

    # Compute median:
    median_ref = median_filter(image, size=3)

    # Wrap compute_error function:
    wrapped_compute_error = jit(nopython=True, parallel=multi_core)(_compute_error)

    for i in range(max_num_iterations):
        # lprint(f"Iteration {i}")

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
        # lprint(f"Number of corrections: {num_corrections}")

        # if no corrections made we stop iterating:
        if num_corrections == 0:
            break

        # We use the median to correct pixels:
        image[mask] = (alpha) * image[mask] + (1 - alpha) * median_ref[mask]

    return image


def _compute_error(array, median, lipschitz):
    # we compute the error map:
    error = median - array
    error = numpy.abs(error)
    error = numpy.maximum(error, lipschitz)
    error -= lipschitz
    return error
