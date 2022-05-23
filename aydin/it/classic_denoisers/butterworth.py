from functools import partial
from typing import Sequence, Union, Optional, Tuple, List

import numpy
from numba import jit
from numpy.fft import fftshift, ifftshift
from numpy.typing import ArrayLike
from scipy.fft import fftn, ifftn
from scipy.special import eval_chebyt

from aydin.it.classic_denoisers import _defaults
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariance import calibrate_denoiser

__fastmath = {'contract', 'afn', 'reassoc'}
__error_model = 'numpy'


def calibrate_denoise_butterworth(
    image: ArrayLike,
    mode: str = 'full',
    axes: Optional[Tuple[int, ...]] = None,
    other_filters: bool = False,
    min_padding: int = 2,
    max_padding: int = 32,
    min_freq: float = 1e-9,
    max_freq: float = 1.0,
    frequency_tolerance: float = 0.05,
    min_order: float = 1.0,
    max_order: float = 8.0,
    crop_size_in_voxels: Optional[int] = _defaults.default_crop_size_superlarge.value,
    optimiser: str = _defaults.default_optimiser.value,
    max_num_evaluations: int = _defaults.default_max_evals_normal.value,
    blind_spots: Optional[List[Tuple[int]]] = _defaults.default_blind_spots.value,
    jinv_interpolation_mode: str = _defaults.default_jinv_interpolation_mode.value,
    multi_core: bool = True,
    display_images: bool = False,
    display_crop: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates the Butterworth denoiser for the given image and returns the optimal
    parameters obtained using the N2S loss.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate Butterworth denoiser for.

    mode: str
        Possible modes are: 'isotropic' for isotropic, meaning only one
        frequency cut-off is calibrated for all axes , 'z-yx' (or 'xy-z') for 3D
        stacks where the cut-off frequency for the x and y axes is the same but
        different for the z axis, 't-z-yx' (or 'xy-z-t') for 3D+t
        timelapse where the cut-off frequency for the x and y axes is the same but
        different for the z and t axis, and 'full' for which all frequency cut-offs are
        different. Use 'z-yx' or 't-z-yx' for axes are ordered as: t, z, y and then x which is the default.
        If for some reason the axis order is reversed you can use 'xy-z' or 'xy-z-t'.
        Note: for 2D+t timelapses where the resolution over x and y are expected to be the same,
        simply use: 'z-yx' (or 'xy-z').

    axes: Optional[Tuple[int,...]]
        Axes over which to apply low-pass filtering.
        (advanced)

    other_filters: bool
        if True other filters similar to Butterworth are tried such as Type II
        Chebyshev filter.

    min_padding: int
        Minimum amount of padding to be added to avoid edge effects.
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

    frequency_tolerance: float
        Frequency tolerance within [-1,1]. A positive value lets more
        high-frequencies to go through the low-pass, negative values bring
        down the cut-off frequencies, zero has no effect. Positive values
        reduces the effect of denoising but also lead to conservative
        filtering that avoids removing potentially important information at
        the boundary (in frequency space) between signal and noise. This can
        also improve the appearance of the images by avoiding the impression
        that the images have been 'over-blurred'. Increase this value by
        small steps of 0.05 to reduce blurring if the image seems too fuzzy.


    min_order: float
        Minimal order for the Butterworth filter to use for calibration.
        (advanced)

    max_order: float
        Maximal order for the Butterworth filter to use for calibration.
        If min_order==max_order then no search is performed for the filter's order.
        (advanced)

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate denoiser.
        Increase this number by factors of two if denoising quality is
        unsatisfactory -- this can be important for very noisy images.
        Values to try are: 256000, 320000, 1'000'000, 2'000'000.
        We do not recommend values higher than 3000000.

    optimiser: str
        Optimiser to use for finding the best denoising
        parameters. Can be: 'smart' (default), or 'fast' for a mix of SHGO
        followed by L-BFGS-B.
        (advanced)

    max_num_evaluations: int
        Maximum number of evaluations for finding the optimal parameters.
        Increase this number by factors of two if denoising quality is
        unsatisfactory.

    blind_spots: Optional[List[Tuple[int]]]
        List of voxel coordinates (relative to receptive field center) to
        be included in the blind-spot. For example, you can give a list of
        3 tuples: [(0,0,0), (0,1,0), (0,-1,0)] to extend the blind spot
        to cover voxels of relative coordinates: (0,0,0),(0,1,0), and (0,-1,0)
        (advanced) (hidden)

    jinv_interpolation_mode: str
        J-invariance interpolation mode for masking. Can be: 'median' or
        'gaussian'.
        (advanced)

    multi_core: bool
        Use all CPU cores during calibration.
        (advanced)

    display_images: bool
        When True the denoised images encountered during optimisation are shown.
        (advanced) (hidden)

    display_crop: bool
        Displays crop, for debugging purposes...
        (advanced) (hidden)

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
    crop = representative_crop(
        image, crop_size=crop_size_in_voxels, display_crop=display_crop
    )

    # Default axes:
    if axes is None:
        axes = tuple(range(image.ndim))

    # ranges:
    freq_cutoff_range = (min_freq, max_freq)
    # order_range = (min_order, max_order)

    # Combine fixed parameters:
    other_fixed_parameters = other_fixed_parameters | {
        'min_padding': min_padding,
        'max_padding': max_padding,
        'axes': axes,
    }

    if mode == 'isotropic' or len(axes) == 1:
        # Partial function:
        _denoise_butterworth = partial(
            denoise_butterworth, **(other_fixed_parameters | {'multi_core': multi_core})
        )

        # Parameters to test when calibrating the denoising algorithm
        parameter_ranges = {'freq_cutoff': freq_cutoff_range}

    elif (mode == 'xy-z' or mode == 'z-yx') and image.ndim == 3:
        # Partial function with parameter impedance match:
        def _denoise_butterworth(*args, **kwargs):
            freq_cutoff_xy = kwargs.pop('freq_cutoff_xy')
            freq_cutoff_z = kwargs.pop('freq_cutoff_z')

            if mode == 'z-yx':
                _freq_cutoff = (freq_cutoff_z, freq_cutoff_xy, freq_cutoff_xy)
            elif mode == 'xy-z':
                _freq_cutoff = (freq_cutoff_xy, freq_cutoff_xy, freq_cutoff_z)

            return denoise_butterworth(
                *args,
                freq_cutoff=_freq_cutoff,
                **(kwargs | other_fixed_parameters | {'multi_core': multi_core}),
            )

        # Parameters to test when calibrating the denoising algorithm
        parameter_ranges = {
            'freq_cutoff_xy': freq_cutoff_range,
            'freq_cutoff_z': freq_cutoff_range,
        }

    elif (mode == 'xy-z-t' or mode == 't-z-yx') and image.ndim == 4:
        # Partial function with parameter impedance match:
        def _denoise_butterworth(*args, **kwargs):
            freq_cutoff_xy = kwargs.pop('freq_cutoff_xy')
            freq_cutoff_z = kwargs.pop('freq_cutoff_z')
            freq_cutoff_t = kwargs.pop('freq_cutoff_t')

            if mode == 't-z-yx':
                _freq_cutoff = (
                    freq_cutoff_t,
                    freq_cutoff_z,
                    freq_cutoff_xy,
                    freq_cutoff_xy,
                )
            elif mode == 'xy-z-t':
                _freq_cutoff = (
                    freq_cutoff_xy,
                    freq_cutoff_xy,
                    freq_cutoff_z,
                    freq_cutoff_t,
                )

            return denoise_butterworth(
                *args,
                freq_cutoff=_freq_cutoff,
                **(kwargs | other_fixed_parameters | {'multi_core': multi_core}),
            )

        # Parameters to test when calibrating the denoising algorithm
        parameter_ranges = {
            'freq_cutoff_xy': freq_cutoff_range,
            'freq_cutoff_z': freq_cutoff_range,
            'freq_cutoff_t': freq_cutoff_range,
        }

    elif mode == 'full':
        # Partial function with parameter impedance match:
        def _denoise_butterworth(*args, **kwargs):
            _freq_cutoff = tuple(
                kwargs.pop(f'freq_cutoff_{i}') for i in range(image.ndim)
            )
            return denoise_butterworth(
                *args,
                freq_cutoff=_freq_cutoff,
                **(kwargs | other_fixed_parameters | {'multi_core': multi_core}),
            )

        # Parameters to test when calibrating the denoising algorithm
        parameter_ranges = {
            f'freq_cutoff_{i}': freq_cutoff_range for i in range(image.ndim)
        }
    else:
        raise ValueError(
            f"Unsupported mode: {mode} for image of dimension: {image.ndim}"
        )

    # This is the code that needs to be uncommented if reverting to single pass:
    # if max_order > min_order:
    #     parameter_ranges |= {'order': order_range}
    # elif max_order == min_order:
    #     other_fixed_parameters |= {'order': min_order}
    # else:
    #     raise ValueError(f"Invalid order range: {min_order} > {max_order}")

    if max_order == min_order:
        other_fixed_parameters |= {'order': min_order}
    elif max_order < min_order:
        raise ValueError(f"Invalid order range: {min_order} > {max_order}")

    if other_filters:
        parameter_ranges |= {'filter_type': ['butterworth', 'chebyshev2']}  #

    # # First optimisation pass:

    # If we only have a single parameter to optimise, we can go for a brute-force approach:
    if len(parameter_ranges) == 1:
        parameter_ranges |= {
            'freq_cutoff': numpy.linspace(
                min_freq, max_freq, max_num_evaluations
            ).tolist()
        }
        # Calibrate denoiser using classic calibrator:
        best_parameters = (
            calibrate_denoiser(
                crop,
                _denoise_butterworth,
                mode=optimiser,
                denoise_parameters=parameter_ranges,
                interpolation_mode=jinv_interpolation_mode,
                blind_spots=blind_spots,
                display_images=display_images,
            )
            | other_fixed_parameters
        )

    else:
        # Calibrate denoiser using smart approach:
        best_parameters = (
            calibrate_denoiser(
                crop,
                _denoise_butterworth,
                mode=optimiser,
                denoise_parameters=parameter_ranges,
                max_num_evaluations=max_num_evaluations,
                interpolation_mode=jinv_interpolation_mode,
                blind_spots=blind_spots,
                display_images=display_images,
            )
            | other_fixed_parameters
        )

    # Second optimisation pass, for the order, if not fixed:
    if max_order > min_order:
        order_list = numpy.linspace(min_order, max_order, 32).tolist()

        parameter_ranges = {'order': order_list} | {
            k: [v] for (k, v) in best_parameters.items()
        }

        best_parameters = (
            calibrate_denoiser(
                crop,
                _denoise_butterworth,
                mode=optimiser,
                denoise_parameters=parameter_ranges,
                max_num_evaluations=max_num_evaluations,
                interpolation_mode=jinv_interpolation_mode,
                blind_spots=blind_spots,
                display_images=display_images,
            )
            | other_fixed_parameters
        )

    # Below we adjust the parameters because denoise_butterworth function (without underscore)
    # uses a different set of parameters...

    if mode == 'isotropic' or len(axes) == 1:
        pass

    elif (mode == 'xy-z' or mode == 'z-yx') and image.ndim == 3:
        # We need to adjust a bit the type of parameters passed to the denoising function:
        freq_cutoff_xy = best_parameters.pop('freq_cutoff_xy')
        freq_cutoff_z = best_parameters.pop('freq_cutoff_z')

        if mode == 'z-yx':
            freq_cutoff = (freq_cutoff_z, freq_cutoff_xy, freq_cutoff_xy)
        elif mode == 'xy-z':
            freq_cutoff = (freq_cutoff_xy, freq_cutoff_xy, freq_cutoff_z)

        best_parameters |= {'freq_cutoff': freq_cutoff}

    elif (mode == 'xy-z-t' or mode == 't-z-yx') and image.ndim == 4:
        # We need to adjust a bit the type of parameters passed to the denoising function:
        freq_cutoff_xy = best_parameters.pop('freq_cutoff_xy')
        freq_cutoff_z = best_parameters.pop('freq_cutoff_z')
        freq_cutoff_t = best_parameters.pop('freq_cutoff_t')

        if mode == 't-z-yx':
            freq_cutoff = (freq_cutoff_t, freq_cutoff_z, freq_cutoff_xy, freq_cutoff_xy)
        elif mode == 'xy-z-t':
            freq_cutoff = (freq_cutoff_xy, freq_cutoff_xy, freq_cutoff_z, freq_cutoff_t)

        best_parameters |= {'freq_cutoff': freq_cutoff}

    elif mode == 'full':
        # We need to adjust a bit the type of parameters passed to the denoising function:
        freq_cutoff = tuple(
            best_parameters.pop(f'freq_cutoff_{i}') for i in range(image.ndim)
        )
        best_parameters |= {'freq_cutoff': freq_cutoff}

    # function to add freq. tol.:
    def _add_freq_tol(f):
        return min(1.0, max(0.0, f + frequency_tolerance))

    # Add frequency_tolerance to all cutoff frequencies:
    if type(best_parameters['freq_cutoff']) is float:
        # If single float we add to freq:
        best_parameters['freq_cutoff'] = _add_freq_tol(best_parameters['freq_cutoff'])
    else:
        # If tuple float we add to all freqs:
        best_parameters['freq_cutoff'] = tuple(
            (_add_freq_tol(f) for f in best_parameters['freq_cutoff'])
        )

    # Memory needed:
    memory_needed = 6 * image.nbytes  # complex numbers and more

    return denoise_butterworth, best_parameters, memory_needed


def denoise_butterworth(
    image: ArrayLike,
    axes: Optional[Tuple[int, ...]] = None,
    filter_type: str = 'butterworth',
    freq_cutoff: Union[float, Sequence[float]] = 0.5,
    order: float = 5,
    min_padding: int = 2,
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

    filter_type: str
        Type of filter. The default is 'butterworth' but we have now introduced
        another filter type: 'chebyshev2' which is a Type II Chebyshev filter
        with a steeper cut-off than Butterworth at the cost of some ripples
        in the rejected high frequencies.
        (advanced)

    freq_cutoff: Union[float, Sequence[float]]
        Single or sequence cutoff frequency, must be within [0, 1]

    order: float
        Filter order, typically an integer above 1.

    min_padding: int
        Minimum amount of padding to be added to avoid edge effects.

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

    # Default axes:
    if axes is None:
        axes = tuple(range(image.ndim))

    # Selected axes:
    selected_axes = tuple((a in axes) for a in range(image.ndim))

    # Normalise freq_cutoff argument to tuple:
    if type(freq_cutoff) is not tuple:
        freq_cutoff = tuple(freq_cutoff if s else 1.0 for s in selected_axes)

    # Number of workers:
    workers = -1 if multi_core else 1

    # First we need to pad the image.
    # By how much? this depends on how much low filtering we need to do:
    pad_width = tuple(
        (
            (
                _apw(s, fc, min_padding, max_padding),
                _apw(s, fc, min_padding, max_padding),
            )
            if sa
            else (0, 0)
        )
        for sa, fc, s in zip(selected_axes, freq_cutoff, image.shape)
    )

    # pad image:
    image = numpy.pad(image, pad_width=pad_width, mode='reflect')

    # Move to frequency space:
    image_f = fftn(image, workers=workers, axes=axes)

    # Center frequencies:
    image_f = fftshift(image_f, axes=axes)

    # Compute squared distance image:
    f = _compute_distance_image(freq_cutoff, image, selected_axes)

    # Choose filter type:
    if filter_type == 'butterworth':
        # Prepare filter:
        _filter = _filter_butterworth
        filter = jit(nopython=True, parallel=multi_core)(_filter)

        # Apply filter:
        image_f = filter(image_f, f, order)

    elif filter_type == 'chebyshev2':
        # Prepare filter:
        _filter = _filter_chebyshev
        filter = jit(nopython=True, parallel=multi_core)(_filter)

        # Apply filter:
        n = 5
        epsilon = 1 / order
        f = numpy.maximum(1e-6, f)
        chebyshev = eval_chebyt(n, 1.0 / f)
        image_f = filter(image_f, epsilon, chebyshev)

    # Shift back:
    image_f = ifftshift(image_f, axes=axes)

    # Back in real space:
    denoised = numpy.real(ifftn(image_f, workers=workers, axes=axes))

    # Crop to remove padding:
    denoised = denoised[
        tuple(
            (slice(u, -v) if (u != 0 and v != 0) else slice(None)) for u, v in pad_width
        )
    ]

    return denoised


# Todo: write a jitted version of this!
# @jit(nopython=True, parallel=True)
def _compute_distance_image(freq_cutoff, image, selected_axes):
    f = numpy.zeros_like(image, dtype=numpy.float32)
    axis_grid = tuple(
        (numpy.linspace(-1, 1, s) if sa else numpy.zeros((s,)))
        for sa, s in zip(selected_axes, image.shape)
    )
    for fc, x in zip(freq_cutoff, numpy.meshgrid(*axis_grid, indexing='ij')):
        f += (x / fc) ** 2
    return f


def _apw(dim_size, freq_cutoff, min_padding, max_padding):
    return min(
        dim_size // 2,
        min(max_padding, max(min_padding, int(1.0 / (1e-10 + freq_cutoff)))),
    )


def _filter_chebyshev(image_f, epsilon, chebyshev):
    image_f /= numpy.sqrt(1 + 1 / ((epsilon * chebyshev) ** 2))
    return image_f


def _filter_butterworth(image_f, f, order):
    image_f /= numpy.sqrt(1 + f ** order)
    return image_f
