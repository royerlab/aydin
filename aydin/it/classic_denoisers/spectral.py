import math
from functools import partial
from typing import Optional, Union, Tuple, Sequence
import numpy
from numba import jit, prange
from numpy.fft import fftshift, ifftshift
from scipy.fft import fftn, ifftn, dctn, idctn, dstn, idstn

from aydin.util.array.outer import outer_sum
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariant_smart import calibrate_denoiser_smart
from aydin.util.patch_size.patch_size import default_patch_size
from aydin.util.patch_transform.patch_transform import (
    extract_patches_nd,
    reconstruct_from_nd_patches,
)


def calibrate_denoise_spectral(
    image,
    axes: Optional[Tuple[int, ...]] = None,
    patch_size: Optional[Union[int, Tuple[int], str]] = None,
    try_dct: bool = True,
    try_fft: bool = True,
    try_dst: bool = False,
    max_order: float = 6.0,
    crop_size_in_voxels: Optional[int] = None,
    max_num_evaluations: int = 256,
    display_images: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates the Spectral denoiser for the given image and returns the optimal
    parameters obtained using the N2S loss.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate spectral denoiser for.

    axes: Optional[Tuple[int,...]]
        Axes over which to apply the spectral transform (dct, dst, fft) for denoising each patch.

    patch_size: int
        Patch size for the 'image-to-patch' transform.
        Can be: 'full' for a single patch covering the whole image, 'half', 'quarter',
        or an int s that corresponds to isotropic patches of shape: (s,)*image.ndim,
        or a tuple of ints. By default (None) the patch size is chosen automatically
        to give the best results.
        (advanced)

    try_dct: bool
        Tries DCT transform during optimisation.

    try_fft: bool
        Tries FFT transform during optimisation.

    try_dst: bool
        Tries DST ransform during optimisation.

    max_order: float
        Maximal order for the Butterworth filter.
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
    crop = representative_crop(image, crop_size=crop_size_in_voxels, display_crop=False)

    # Normalise patch size:
    patch_size = default_patch_size(image, patch_size, odd=True)

    # Ranges:
    threshold_range = (0.0, 1.0)  # np.arange(0, 0.5, 0.02) ** 2
    freq_bias_range = (0.0, 2.0)  # np.arange(0, 2, 0.2)
    freq_cutoff_range = (0.01, 1.0)
    order_range = (0.5, max_order)

    # prepare modes list
    modes = []
    if try_dct:
        modes.append("dct")
    if try_fft:
        modes.append("fft")
    if try_dst:
        modes.append("dst")

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {
        'threshold': threshold_range,
        'freq_bias_stength': freq_bias_range,
        # 'reconstruction_gamma': [0.0001, 0.1, 1.0],
        'freq_cutoff': freq_cutoff_range,
        'order': order_range,
        'mode': modes,
    }  # 'fft',, 'fft'

    # Combine fixed parameters:
    other_fixed_parameters = other_fixed_parameters | {
        'patch_size': patch_size,
        'axes': axes,
    }

    # Partial function:
    _denoise_spectral = partial(
        denoise_spectral, **(other_fixed_parameters | {'multi_core': False})
    )

    # Calibrate denoiser
    best_parameters = (
        calibrate_denoiser_smart(
            crop,
            _denoise_spectral,
            denoise_parameters=parameter_ranges,
            display_images=display_images,
            max_num_evaluations=max_num_evaluations,
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = 2 * image.nbytes + 8 * image.nbytes * math.prod(patch_size)

    return denoise_spectral, best_parameters, memory_needed


def denoise_spectral(
    image,
    axes: Optional[Tuple[int, ...]] = None,
    patch_size: Optional[Union[int, Tuple[int], str]] = None,
    mode: str = 'dct',
    threshold: float = 0.5,
    freq_bias_stength: float = 1,
    freq_cutoff: Union[float, Sequence[float]] = 0.5,
    order: float = 1,
    reconstruction_gamma: float = 0,
    multi_core: bool = True,
):
    """Denoises the given image by first applying the patch
    transform, and then zeroing Fourier/DCT/DST coefficients
    below a given threshold. In addition, we apply Butterworth
    filter to suppress frequencies above the band-pass and a
    configurable frequency bias before applying the thresholding
    to favour suppressing high versus low frequencies.
    \n\n
    Note: This seems like a lot of parameters, but thanks to our
    auto-tuning approach these parameters are all automatically
     determined 😊.

    Parameters
    ----------
    image: ArrayLike
        Image to denoise

    axes: Optional[Tuple[int,...]]
        Axes over which to apply the spetcral transform (dct, dst, fft) for denoising each patch.

    patch_size: int
        Patch size for the 'image-to-patch' transform.
        Can be: 'full' for a single patch covering the whole image, 'half', 'quarter',
        or an int s that corresponds to isotropic patches of shape: (s,)*image.ndim,
        or a tuple of ints. By default (None) the patch size is chosen automatically
        to give the best results.

    mode: str
        Possible modes are: 'dct'(works best!), 'dst', and 'fft'.

    threshold: float
        Threshold between 0 and 1

    freq_bias_stength: float
        Frequency bias: closer to zero: no bias against high frequencies,
        closer to one and above: stronger bias towards high-frequencies.

    freq_cutoff: float
        Cutoff frequency, must be within [0, 1]. In addition

    order: float
        Filter order, typically an integer above 1.

    reconstruction_gamma: float
        Patch reconstruction parameter

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

    # 'full' patch size:
    if patch_size == 'full':
        patch_size = image.shape
    elif patch_size == 'half':
        patch_size = tuple(max(3, 2 * (s // 4)) for s in image.shape)
    elif patch_size == 'quarter':
        patch_size = tuple(max(3, 4 * (s // 8)) for s in image.shape)

    # Normalise patch size:
    patch_size = default_patch_size(image, patch_size, odd=True)

    # Default axes:
    if axes is None:
        axes = tuple(range(image.ndim))

    # Selected axes:
    selected_axes = tuple((a in axes) for a in range(image.ndim))

    workers = -1 if multi_core else 1
    axes = tuple(a for a in range(1, image.ndim + 1) if (a - 1) in axes)
    if mode == 'fft':
        transform = lambda x: fftshift(  # noqa: E731
            fftn(x, workers=workers, axes=axes), axes=axes
        )
        i_transform = lambda x: ifftn(  # noqa: E731
            ifftshift(x, axes=axes), workers=workers, axes=axes
        )
    elif mode == 'dct':
        transform = partial(dctn, workers=workers, axes=axes)
        i_transform = partial(idctn, workers=workers, axes=axes)
    elif mode == 'dst':
        transform = partial(dstn, workers=workers, axes=axes)
        i_transform = partial(idstn, workers=workers, axes=axes)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Normalise freq_cutoff argument to tuple:
    if type(freq_cutoff) is not tuple:
        freq_cutoff = tuple((freq_cutoff,) * image.ndim)

    # First we apply the patch transform:
    patches = extract_patches_nd(image, patch_size=patch_size)

    # ### PART 1: apply Butterworth filter to patches:

    # Then we apply the sparsifying transform:
    patches = transform(patches)

    # Compute adequate squared distance image and chose filter implementation:
    if mode == 'fft':
        f = _compute_distance_image_for_fft(freq_cutoff, patch_size, selected_axes)
    elif mode == 'dct' or mode == 'dst':
        f = _compute_distance_image_for_dxt(freq_cutoff, patch_size, selected_axes)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Configure filter function:
    filter_wrapped = jit(nopython=True, parallel=multi_core)(_filter)

    # Apply filter:
    patches = filter_wrapped(patches, f, order)

    # ### PART 2: apply thresholding:

    # Window for frequency bias:
    freq_bias = _freq_bias_window(patch_size, freq_bias_stength)

    # We use this value to estimate power per coefficient:
    power = numpy.absolute(patches)
    power *= freq_bias

    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(image, name='image')
    #     viewer.add_image(patches, name='patches')
    #     viewer.add_image(f_patches, name='f_patches')
    #     viewer.add_image(power, name='power')
    #     viewer.add_image(freq_bias, name='freq_bias')

    # What is the max coefficient in the transforms:
    max_value = numpy.max(power)

    # We derive from that the threshold:
    threshold *= max_value

    # Here are the entries that are below the threshold:
    below = power < threshold

    # Thresholding:
    patches[below] = 0

    # Transform back to real space:
    patches = i_transform(patches)

    # convert to real:
    if numpy.iscomplexobj(patches):
        patches = numpy.real(patches)

    # Transform back from patches to image:
    denoised_image = reconstruct_from_nd_patches(
        patches, image.shape, gamma=reconstruction_gamma
    )

    # Cast back to float32 if needed:
    denoised_image = denoised_image.astype(numpy.float32, copy=False)

    return denoised_image


def _freq_bias_window(shape: Tuple[int], alpha: float = 1):
    window_tuple = tuple(numpy.linspace(0, 1, s) ** 2 for s in shape)
    window_nd = numpy.sqrt(outer_sum(*window_tuple)) + 1e-6
    window_nd = 1.0 / (1.0 + window_nd)
    window_nd **= alpha
    window_nd /= window_nd.max()
    window_nd = window_nd.astype(numpy.float32)
    return window_nd


def _compute_distance_image_for_dxt(freq_cutoff, shape, selected_axes):

    # Normalise selected axes:
    if selected_axes is None:
        selected_axes = (a for a in range(len(shape)))

    f = numpy.zeros(shape=shape, dtype=numpy.float32)
    axis_grid = tuple(
        (numpy.linspace(0, 1, s) if sa else numpy.zeros((s,)))
        for sa, s in zip(selected_axes, shape)
    )
    for fc, x in zip(freq_cutoff, numpy.meshgrid(*axis_grid, indexing='ij')):
        f += (x / fc) ** 2
    return f


def _compute_distance_image_for_fft(freq_cutoff, shape, selected_axes):
    f = numpy.zeros(shape=shape, dtype=numpy.float32)
    axis_grid = tuple(
        (numpy.linspace(-1, 1, s) if sa else numpy.zeros((s,)))
        for sa, s in zip(selected_axes, shape)
    )
    for fc, x in zip(freq_cutoff, numpy.meshgrid(*axis_grid, indexing='ij')):
        f += (x / fc) ** 2
    return f


def _filter(image_f, f, order):
    factor = (1.0 + numpy.sqrt(f) ** (2 * order)) ** (-0.5)
    factor = factor.astype(numpy.float32)
    n = image_f.shape[0]
    for i in prange(n):
        image_f[i] *= factor
    return image_f
