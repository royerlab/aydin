from functools import partial
from typing import Sequence, Union, Optional, Tuple

import numpy
from numba import jit
from numpy.fft import fftshift, ifftshift
from scipy.fft import fftn, ifftn

from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariant_smart import calibrate_denoiser_smart

__fastmath = {'contract', 'afn', 'reassoc'}
__error_model = 'numpy'


def calibrate_denoise_butterworth(
    image,
    isotropic: bool = False,
    axes: Optional[Tuple[int, ...]] = None,
    max_padding: int = 32,
    min_freq: float = 0.001,
    max_freq: float = 1.0,
    min_order: float = 0.5,
    max_order: float = 6.0,
    crop_size_in_voxels: Optional[int] = 128000,
    max_num_evaluations: int = 512,
    display_images: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates the Butterworth denoiser for the given image and returns the optimal
    parameters obtained using the N2S loss.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate Sobolev denoiser for.

    isotropic: bool
        When True, the filtering is isotropic
        i.e. all frequency cutoffs are the same along all axis,
        but when false, the frequency cutoffs are different for different axis.
        Anisotropic filtering is usefull for example for 3D microscopy images
        that have a different resolution along z than along x and y, or for nD+t
        images that have a very different correlation structure along time than
        along space.

    axes: Optional[Tuple[int,...]]
        Axes over which to apply low-pass filtering.
        (advanced)

    max_padding: int
        Maximum amount of padding to be added to avoid edge effects.
        (advanced)

    min_freq: float
        Minimum cutoff frequency to use for calibration. Must be within [0,1],
        typically close to zero.
        (advanced)

    max_freq: float
        Maximum cutoff frequency to use for calibration. Must be within [0,1],
        typically close to one.
        (advanced)

    max_order: float
        Maximal order for the Butterworth filter to use for calibration.
        (advanced)

    max_order: float
        Minimal order for the Butterworth filter to use for calibration.
        (advanced)

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate denoiser.
        (advanced)

    max_num_evaluations: int
        Maximum number of evaluations for finding the optimal parameters.
        (advanced)

    display_images: bool
        When True the denoised images encountered during optimisation are shown.
        (advanced)

    other_fixed_parameters: dict
        Any other fixed parameters. (advanced)

    Returns
    -------
    Denoising function, dictionary containing optimal parameters,
    and free memory needed in bytes for computation.
    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # obtain representative crop, to speed things up...
    crop = representative_crop(image, crop_size=crop_size_in_voxels)

    # ranges:
    freq_cutoff_range = (min_freq, max_freq)
    order_range = (min_order, max_order)

    # Combine fixed parameters:
    other_fixed_parameters = other_fixed_parameters | {
        'max_padding': max_padding,
        'axes': axes,
    }

    if isotropic:
        # Partial function:
        _denoise_butterworth = partial(
            denoise_butterworth, **(other_fixed_parameters | {'multi_core': False})
        )

        # Parameters to test when calibrating the denoising algorithm
        parameter_ranges = {'freq_cutoff': freq_cutoff_range, 'order': order_range}
    else:
        # Partial function with parameter impedance match:
        def _denoise_butterworth(*args, **kwargs):
            _freq_cutoff = tuple(
                kwargs.pop(f'freq_cutoff_{i}') for i in range(image.ndim)
            )
            return denoise_butterworth(
                *args,
                freq_cutoff=_freq_cutoff,
                **(kwargs | other_fixed_parameters | {'multi_core': False}),
            )

        # Parameters to test when calibrating the denoising algorithm
        parameter_ranges = {
            f'freq_cutoff_{i}': freq_cutoff_range for i in range(image.ndim)
        } | {'order': order_range}

    # Calibrate denoiser
    best_parameters = (
        calibrate_denoiser_smart(
            crop,
            _denoise_butterworth,
            denoise_parameters=parameter_ranges,
            max_num_evaluations=max_num_evaluations,
            display_images=display_images,
        )
        | other_fixed_parameters
    )

    if not isotropic:
        # We need to adjust a bit the type of parameters passed to the denoising function:
        freq_cutoff = tuple(
            best_parameters.pop(f'freq_cutoff_{i}') for i in range(image.ndim)
        )
        best_parameters |= {'freq_cutoff': freq_cutoff}

    # Memory needed:
    memory_needed = 6 * image.nbytes  # complex numbers and more

    return denoise_butterworth, best_parameters, memory_needed


def denoise_butterworth(
    image,
    axes: Optional[Tuple[int, ...]] = None,
    freq_cutoff: Union[float, Sequence[float]] = 0.5,
    order: float = 1,
    max_padding: int = 32,
    multi_core: bool = True,
):
    """
    Denoises the given image by applying a configurable <a
    href="https://en.wikipedia.org/wiki/Butterworth_filter">Butterworth
    lowpass filter</a>. Remarkably good when your signal
    does not have high-frequencies beyond a certain cutoff frequency.
    This is probably the first algorithm that should be tried of all
    currently available in Aydin. It is actually quite impressive how
    well this performs in practice. If the signal in your images is
    band-limited as is often the case for microscopy images, this
    denoiser will work great.
    \n\n
    Note: We recommend applying a variance stabilisation transform
    to improve results for images with non-Gaussian noise.

    Parameters
    ----------
    image: ArrayLike
        Image to be denoised

    axes: Optional[Tuple[int,...]]
        Axes over which to apply lowpass filtering.

    freq_cutoff: float
        Cutoff frequency, must be within [0, 1]

    order: float
        Filter order, typically an integer above 1.

    max_padding: int
        Maximum amount of padding to be added to avoid edge effects.

    multi_core: bool
        By default we use as many cores as possible, in some cases, for small
        (test) images, it might be faster to run on a single core instead of
        starting the whole parallelization machinery.


    Returns
    -------
    Denoised image

    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # Normalise freq_cutoff argument to tuple:
    if type(freq_cutoff) is not tuple:
        freq_cutoff = tuple((freq_cutoff,) * image.ndim)

    # Number of workers:
    workers = -1 if multi_core else 1

    # Default axes:
    if axes is None:
        axes = tuple(range(image.ndim))

    # Selected axes:
    selected_axes = tuple((a in axes) for a in range(image.ndim))

    # First we need to pad the image.
    # By how much? this depends on how much low filtering we need to do:
    pad_width = tuple(
        ((_apw(fc, max_padding), _apw(fc, max_padding)) if sa else (0, 0))
        for sa, fc in zip(selected_axes, freq_cutoff)
    )

    # pad image:
    image = numpy.pad(image, pad_width=pad_width, mode='reflect')

    # Move to frequency space:
    image_f = fftn(image, workers=workers, axes=axes)

    # Center frequencies:
    image_f = fftshift(image_f, axes=axes)

    # Compute squared distance image:
    f = _compute_distance_image(freq_cutoff, image, selected_axes)

    # Chose filter implementation:
    filter = jit(nopython=True, parallel=multi_core)(_filter)

    # Apply filter:
    image_f = filter(image_f, f, order)

    # Shift back:
    image_f = ifftshift(image_f, axes=axes)

    # Back in real space:
    denoised = numpy.real(ifftn(image_f, workers=workers, axes=axes))

    # Crop to remove padding:
    denoised = denoised[tuple(slice(u, -v) for u, v in pad_width)]

    return denoised


def _compute_distance_image(freq_cutoff, image, selected_axes):
    f = numpy.zeros_like(image, dtype=numpy.float32)
    axis_grid = tuple(
        (numpy.linspace(-1, 1, s) if sa else numpy.zeros((s,)))
        for sa, s in zip(selected_axes, image.shape)
    )
    for fc, x in zip(freq_cutoff, numpy.meshgrid(*axis_grid, indexing='ij')):
        f += (x / fc) ** 2
    return f


def _apw(freq_cutoff, max_padding):
    return min(max_padding, max(1, int(1.0 / (1e-10 + freq_cutoff))))


def _filter(image_f, f, order):
    image_f *= (1 + numpy.sqrt(f) ** (2 * order)) ** (-0.5)
    return image_f
