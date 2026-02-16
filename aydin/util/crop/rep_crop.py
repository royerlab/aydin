"""Representative crop extraction from n-dimensional images.

Finds and extracts the most informative sub-region of an image by
scoring candidate crops based on contrast or edge magnitude metrics.
"""

import math
import time
from random import randrange
from typing import Callable, Optional

import numpy
from numba import jit, prange
from numpy import absolute
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter, sobel

from aydin.util.edge_filter.fast_edge_filter import fast_edge_filter
from aydin.util.log.log import aprint, asection


def representative_crop(
    image: ArrayLike,
    mode: str = 'contrast',
    crop_size: Optional[int] = None,
    min_length: int = 32,
    smoothing_sigma: float = 0.5,
    equal_sides: bool = False,
    favour_odd_lengths: bool = False,
    search_mode: str = 'random',
    granularity_factor: int = 4,
    random_search_mode_num_crops: int = 1512,
    min_num_crops: int = 512,
    timeout_in_seconds: float = 2,
    return_slice: bool = False,
    display_crop: bool = False,
    std_fun: Callable = None,  # numpy.std
):
    """Extract a representative crop from the image. Searches for the crop of given
    (approximate) size with highest score. The score is simply the sum of sobel
    magnitudes (~tenengrad) which is a good metric for estimating how much interesting
    content each crop contains. Empirically, this highly correlates with where
    I (Loic A. Royer) tend to look at in images.


    Parameters
    ----------
    image : ArrayLike
        Image to extract representative crop from

    mode : str
        Metric for picking crop. Can be : 'contrast' (fastest), 'sobel', 'sobelmin',
        'sobelmax' We recommend 'contrast'.

    crop_size : int
        Crop size in voxels. Default (None) is 32000.

    min_length : int
        Crop axis lengths cannot be smaller than this number.

    smoothing_sigma : float
        Sigma value for Gaussian filter smoothing to achieve some crude denoising and thus
        make it a bit easier to estimate the score per crop.

    equal_sides : bool
        When True the crop will have all its sides of equal size (square, cube, ...)

    favour_odd_lengths : bool
        If possible favours crops that have odd shape lengths.

    search_mode : str
        Search mode for best crops. Can be 'random' or 'systematic'. In
        random mode we pick random crops, in systematic mode we check every
        possible strided crop.

    granularity_factor : int
        Granularity of search. Higher values correspond to more overlap between candidate crops.

    random_search_mode_num_crops : int
        Number of crops to check in 'random' search mode.

    min_num_crops : int
        Min number of crops to examine.

    timeout_in_seconds : float
        Maximum amount of time in seconds that this function should run for.
        This avoids excessive computation for very large images.

    return_slice : bool
        If True the slice is returned too.

    display_crop : bool
        Displays crop, for debugging purposes.

    std_fun : callable, optional
        Function to compute the contrast score for a crop.
        Defaults to a fast parallel standard deviation estimator.

    Returns
    -------
    numpy.ndarray or tuple
        The most representative crop. If ``return_slice`` is True,
        returns a tuple of (crop, slice_tuple).
    """

    # Debug:
    # _fast_std.parallel_diagnostics(level=4)

    # Std function:
    if std_fun is None:
        std_fun = _fast_std

    # Compile numba functions:
    # _fast_std(numpy.ones(100, dtype=numpy.float32))
    # _normalise(numpy.ones(100, dtype=numpy.float32))

    # Start time:
    start_time = time.time()

    with asection(
        f"Cropping image of size: {image.shape} with at most {crop_size} voxels and mode {mode}"
    ):

        # save reference to original image:
        original_image = image

        with asection("Cast and normalise image..."):
            # Cast, if needed:
            image = image.astype(numpy.float32, copy=False)
            # Normalise:
            # image = _normalise(image)

        # Apply filter:
        with asection(f"Apply cropping filter to image of shape: {image.shape}"):

            # Smoothing:
            sigma = tuple(
                (smoothing_sigma if s > min_length else 0 for s in image.shape)
            )
            image = gaussian_filter(image, sigma=sigma)

            if mode == 'contrast':
                pass
            elif mode == 'sobelfast':
                image = _sobel_fast(image)
            elif mode == 'sobel':
                image = _sobel_magnitude(image)
            elif mode == 'sobelmin':
                image = _sobel_minimum(image)
            elif mode == 'sobelmax':
                image = _sobel_maximum(image)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        #
        # import napari
        # with napari.gui_qt():
        #     viewer = napari.Viewer()
        #     viewer.add_image(image, name='image')
        #     viewer.add_image(filtered_image, name='filtered_image')

        # To speed up cropping we apply a divide and conquer to 'pre-crop' the image recusively:
        # if precropping:
        #     image = _precrop(image,
        #                      crop_size=crop_size,
        #                      min_length=min_length,
        #                      mode=mode,
        #                      smoothing_size=smoothing_size)

        # Number of voxels in image:
        num_voxels = image.size

        # Default number of voxels:
        if crop_size is None:
            crop_size = 32000

        # Ratio by which to crop to achieve max num voxels:
        ratio = (crop_size / num_voxels) ** (1 / image.ndim)

        if ratio >= 1:
            # If the image is small enough no point in getting a crop!
            if return_slice:
                return image, (slice(None),) * image.ndim
            else:
                return image

        # cropped shape:
        cropped_shape = tuple(
            min(max(min_length, int(s * ratio)), s) for s in image.shape
        )

        # If the crop size is still too big, we adjust that. This happens because
        # we cannot crop dimensions that are too small, leading to an
        # underestimation of the ratio.
        for tries in range(8):
            # First let's figure out the current crop size:
            current_crop_size = math.prod(cropped_shape)

            # we check if it is ok, or too large:
            if current_crop_size < 1.05 * crop_size:
                # we are ok if the crop size is within 5% of the desired size.
                break

            # If too large we compute the ratio by which to adjust it:
            ratio = (crop_size / current_crop_size) ** (1 / image.ndim)

            # we compute a new crop shape:
            cropped_shape = tuple(
                min(max(min_length, int(s * ratio)), s) for s in cropped_shape
            )

        # Favour odd lengths if requested:
        if favour_odd_lengths:
            cropped_shape = tuple((s // 2) * 2 + 1 for s in cropped_shape)

        # We enforce equal sides if requested:
        if equal_sides:
            min_length = min(cropped_shape)
            cropped_shape = tuple((min_length,) * image.ndim)

        # range for translation:
        translation_range = tuple(s - cs for s, cs in zip(image.shape, cropped_shape))

        # We loop through a number of crops and keep the one wit the best score:
        best_score = -1
        best_slice = None
        best_crop = None

        # Instead of searching for all possible crops, we take into
        # account the size of the crops to define a 'granularity' (
        # stride) of the translations used for search:

        granularity = tuple(max(1, cs // granularity_factor) for cs in cropped_shape)

        if search_mode == 'random':

            # We make sure that the number of crops is not too large given
            # the relative size of the crop versus whole image:
            random_search_mode_num_crops = min(
                random_search_mode_num_crops,
                (granularity_factor**image.ndim) * int(image.size / crop_size),
            )

            for i in range(random_search_mode_num_crops):

                # translation:
                translation = tuple(
                    (randrange(0, max(1, (s - cs) // g)) * g if cs != s else 0)
                    for s, cs, g in zip(image.shape, cropped_shape, granularity)
                )

                # function to get crop slice:
                def _crop_slice(translation, cropped_shape, downscale: int = 1):
                    """Build a slice tuple for cropping at a given translation."""
                    return tuple(
                        slice(t, t + s, downscale)
                        for t, s in zip(translation, cropped_shape)
                    )

                # slice object for cropping:
                crop_slice = _crop_slice(
                    translation, cropped_shape, 2 if image.size > 1e8 else 1
                )

                # extract crop:
                crop = image[crop_slice]

                score = std_fun(crop)

                # slice object for the actual crop:
                crop_slice = _crop_slice(translation, cropped_shape, 1)

                # update best score and image:
                if score > best_score and not math.isinf(score):
                    best_score = score
                    best_slice = crop_slice

                    # We make sure to have the full and original crop!
                    best_crop = original_image[best_slice]

                if i >= min_num_crops and time.time() > start_time + timeout_in_seconds:
                    aprint(
                        f"Interrupting crop search because of timeout after {i} crops examined!"
                    )
                    break

        elif search_mode == 'systematic':

            # grid for translations:
            translation_indices = tuple(
                max(1, int(granularity_factor * r / cs))
                for r, cs in zip(translation_range, cropped_shape)
            )

            for i, index in enumerate(numpy.ndindex(translation_indices)):

                # print(
                #     f"i={i}, index={index}, translation_indices={translation_indices}"
                # )

                # translation:
                translation = tuple(
                    int(j * cs / granularity_factor)
                    for j, cs in zip(index, cropped_shape)
                )

                # slice object for cropping:
                crop_slice = tuple(
                    slice(t, t + s) for t, s in zip(translation, cropped_shape)
                )

                # extract crop:
                crop = image[crop_slice]

                score = std_fun(crop)

                # update best score and image:
                if score > best_score and not math.isinf(score):
                    best_score = score
                    best_slice = crop_slice

                    # We make sure to have the full and original crop!
                    best_crop = original_image[best_slice]

                if i >= min_num_crops and time.time() > start_time + timeout_in_seconds:
                    aprint(
                        f"Interrupting crop search because of timeout after {i} crops examined!"
                    )
                    break
        else:
            raise ValueError(f"Unsupported search mode: {search_mode}")

        aprint(f"Selected crop: shape={best_crop.shape}, score={best_score:.4f}")

        if display_crop:

            import napari

            viewer = napari.Viewer()
            viewer.add_image(image.squeeze(), name='image')
            viewer.add_image(best_crop.squeeze(), name='best_crop')
            napari.run()

        #     print(_fast_std.signatures)
        #     for sig in _fast_std.signatures:
        #         print(_fast_std.inspect_asm(sig))`

        if return_slice:
            return best_crop, best_slice
        else:
            return best_crop


def _sobel_magnitude(image):
    """Compute the Sobel gradient magnitude across all large-enough axes.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array.

    Returns
    -------
    numpy.ndarray
        Root-sum-of-squares of per-axis Sobel gradients.
    """
    magnitude = numpy.zeros_like(image)
    for axis in range(image.ndim):
        if image.shape[axis] < 32:
            continue
        magnitude += sobel(image, axis=axis) ** 2
    return numpy.sqrt(magnitude)


def _sobel_minimum(image):
    """Compute the element-wise minimum of absolute Sobel gradients across axes.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array.

    Returns
    -------
    numpy.ndarray
        Per-pixel minimum of absolute Sobel gradients.
    """
    minimum = None
    for axis in range(image.ndim):
        if image.shape[axis] < 32:
            continue
        sobel_image = absolute(sobel(image, axis=axis))
        if minimum is None:
            minimum = sobel_image
        else:
            minimum = numpy.minimum(minimum, sobel_image)
    if minimum is None:
        return image
    else:
        return minimum


def _sobel_maximum(image):
    """Compute the element-wise maximum of absolute edge filter responses across axes.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array.

    Returns
    -------
    numpy.ndarray
        Per-pixel maximum of absolute edge filter responses.
    """
    maximum = None
    for axis in range(image.ndim):
        if image.shape[axis] < 32:
            continue
        sobel_image = absolute(fast_edge_filter(image, axis=axis))
        if maximum is None:
            maximum = sobel_image
        else:
            maximum = numpy.maximum(maximum, sobel_image)
    if maximum is None:
        return image
    else:
        return maximum


def _sobel_fast(image):
    """Compute an absolute edge filter along the longest axis only.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array.

    Returns
    -------
    numpy.ndarray
        Absolute edge filter response along the longest axis.
    """
    longest_axis = max(image.shape)
    axis = image.shape.index(longest_axis)

    return absolute(fast_edge_filter(image, axis=axis))


def _normalise(image):
    """Normalise an image to the [0, 1] range using min-max scaling.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array of float32 values.

    Returns
    -------
    numpy.ndarray
        Normalised image with values in [0, 1].
    """
    # Normalise:
    image_min = numpy.float32(image.min())
    image_max = numpy.float32(image.max())
    if image_max - image_min > 0:
        return _rescale(image, image_min, image_max)
    else:
        return (image - image_min).astype(numpy.float32)


def _rescale(x, min_value, max_value):
    """Rescale values from [min_value, max_value] to [0, 1].

    Parameters
    ----------
    x : array_like
        Values to rescale.
    min_value : float
        Minimum of the original range.
    max_value : float
        Maximum of the original range.

    Returns
    -------
    numpy.ndarray
        Rescaled values in [0, 1], dtype float32.
    """
    return ((x - min_value) / (max_value - min_value)).astype(numpy.float32)


@jit(nopython=True, parallel=True, fastmath=True)
def _fast_std(image: ArrayLike, workers=16, decimation=1):
    """Compute a fast approximate standard deviation using parallel chunks.

    Splits the flattened array into chunks and returns the maximum
    per-chunk standard deviation, providing a fast lower bound on
    local contrast.

    Parameters
    ----------
    image : ArrayLike
        Input image array.
    workers : int, optional
        Number of parallel chunks to process, by default 16.
    decimation : int, optional
        Decimation factor for sub-sampling within each chunk, by default 1.

    Returns
    -------
    float
        Maximum standard deviation across all chunks.
    """
    array = image.ravel()
    length = array.size
    num_chunks = workers
    chunk_length = (length // num_chunks) + num_chunks

    std = 0.0

    for c in prange(num_chunks):

        start = c * chunk_length
        stop = (c + 1) * chunk_length
        if stop >= length:
            stop = length

        sub_array = array[start:stop:decimation]
        chunk_std = numpy.std(sub_array)
        std = max(std, chunk_std)

    return std
