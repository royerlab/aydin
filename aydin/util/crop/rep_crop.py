import math
import time
from random import randrange
from typing import Optional

import numpy
from numpy import absolute
from numpy.typing import ArrayLike
from scipy.ndimage import sobel, gaussian_filter

from aydin.util.edge_filter.fast_edge_filter import fast_edge_filter
from aydin.util.fast_uniform_filter.parallel_uf import parallel_uniform_filter
from aydin.util.log.log import lprint, lsection


def representative_crop(
    image: ArrayLike,
    mode: str = 'contrast',
    crop_size: Optional[int] = None,
    min_length: int = 32,
    smoothing_sigma: int = 0.5,
    equal_sides: bool = False,
    favour_odd_lengths: bool = False,
    search_mode: str = 'random',
    random_search_mode_num_crops: int = 1500,
    timeout_in_seconds: float = 3,
    return_slice: bool = False,
    display_crop: bool = False,
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
        Strategy for picking crop.

    crop_size : int
        Crop size in voxels. Default (None) is 32000.

    min_length : int
        Crop axis lengths cannot be smaller than this number.

    smoothing_sigma : int
        Sigma value for Gaussian filter smoothing to achieve some crude denoising and thus
        make it a bit easier to estimate the score per crop.

    equal_sides : bool
        When True the crop will have all its sides of equal size (square, cube, ...)

    favour_odd_lengths : bool
        If possible favours crops that have odd shape lengths.

    search_mode: bool
        Search mode for best crops. Can be 'random' or 'systematic'. In
        random mode we pick random crops, in systematic mode we check every
        possible strided crop.

    random_search_mode_num_crops: int
        Number of crops to check in 'random' search mode.

    timeout_in_seconds: float
        Maximum amount of time in seconds that this function should run for.
        This avoids excessive computation for very large images.

    return_slice : bool
        If True the slice is returned too:

    display_crop: bool
        Displays crop, for debugging purposes...

    Returns
    -------
    Most representative crop, and if return_slice is True the actual slice object too.

    """

    # Start time:
    start_time = time.time()

    # convert type:
    image = image.astype(dtype=numpy.float32, copy=False)

    # Normalise:
    image_min = image.min()
    image_max = image.max()
    image -= image_min
    if image_max > 0:
        image /= image_max

    # Apply filter:
    with lsection(f"Apply cropping filter to image of: {image.shape}"):

        sigma = tuple((smoothing_sigma if s > min_length else 0 for s in image.shape))
        size = tuple((16 if s > min_length else 1 for s in image.shape))

        filtered_image = gaussian_filter(image, sigma=sigma) - parallel_uniform_filter(
            image, size=size
        )

        if mode == 'contrast':
            pass
        elif mode == 'sobelfast':
            filtered_image = _sobel_fast(filtered_image)
        elif mode == 'sobel':
            filtered_image = _sobel_magnitude(filtered_image)
        elif mode == 'sobelmin':
            filtered_image = _sobel_minimum(filtered_image)
        elif mode == 'sobelmax':
            filtered_image = _sobel_maximum(filtered_image)
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
        return image

    # cropped shape:
    cropped_shape = tuple(min(max(min_length, int(s * ratio)), s) for s in image.shape)

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
    granularity_factor = 4
    granularity = tuple(cs // granularity_factor for cs in cropped_shape)

    if search_mode == 'random' or image.size > 1e6:

        # We make sure that the number of crops is not too large given
        # the relative size of the crop versus whole image:
        random_search_mode_num_crops = min(
            random_search_mode_num_crops,
            (granularity_factor**image.ndim) * int(image.size / crop_size),
        )

        for index in range(random_search_mode_num_crops):

            # translation:
            translation = tuple(
                (randrange(0, max(1, (s - cs) // g)) * g if cs != s else 0)
                for s, cs, g in zip(image.shape, cropped_shape, granularity)
            )

            # function to get crop slice:
            def _crop_slice(translation, cropped_shape, downscale: int = 1):
                return tuple(
                    slice(t, t + s, downscale)
                    for t, s in zip(translation, cropped_shape)
                )

            # slice object for cropping:
            crop_slice = _crop_slice(
                translation, cropped_shape, 2 if image.size > 1e8 else 1
            )

            # extract crop:
            crop = filtered_image[crop_slice]

            score = numpy.std(crop)

            # slice object for the actual crop:
            crop_slice = _crop_slice(translation, cropped_shape, 1)

            # update best score and image:
            if score > best_score and not math.isinf(score):
                best_score = score
                best_slice = crop_slice

                # We make sure to have the full and original crop!
                best_crop = image[best_slice]

            if time.time() > start_time + timeout_in_seconds:
                lprint(
                    f"Interrupting crop search because of timeout after {index} crops examined!"
                )
                break

    elif search_mode == 'systematic':

        # grid for translations:
        translation_indices = tuple(
            max(1, int(granularity_factor * r / cs))
            for r, cs in zip(translation_range, cropped_shape)
        )

        for i, index in enumerate(numpy.ndindex(translation_indices)):

            # translation:
            translation = tuple(
                int(i * cs / granularity_factor) for i, cs in zip(index, cropped_shape)
            )

            # slice object for cropping:
            crop_slice = tuple(
                slice(t, t + s) for t, s in zip(translation, cropped_shape)
            )

            # extract crop:
            crop = filtered_image[crop_slice]

            score = numpy.std(crop)

            # update best score and image:
            if score > best_score and not math.isinf(score):
                best_score = score
                best_slice = crop_slice

                # We make sure to have the full and original crop!
                best_crop = image[best_slice]

            if time.time() > start_time + timeout_in_seconds:
                lprint(
                    f"Interrupting crop search because of timeout after {i} crops examined!"
                )
                break
    else:
        raise ValueError(f"Unsupported search mode: {search_mode}")

    if display_crop:

        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name='image')
            viewer.add_image(best_crop, name='best_crop')

    if return_slice:
        return best_crop, best_slice
    else:
        return best_crop


def _sobel_magnitude(image):
    magnitude = numpy.zeros_like(image)
    for axis in range(image.ndim):
        if image.shape[axis] < 32:
            continue
        magnitude += sobel(image, axis=axis) ** 2
    return numpy.sqrt(magnitude)


def _sobel_minimum(image):
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

    longest_axis = max(image.shape)
    axis = image.shape.index(longest_axis)

    return absolute(fast_edge_filter(image, axis=axis))


#
# def _precrop(image, crop_size, min_length, mode, smoothing_size, slice=None):
#
#     # If image is already smaller than requested then we don't need to do anything:
#     if image.size <= crop_size:
#         return image
#
#     # cropped shape:
#     cropped_shape = tuple(
#         min(max(min_length, int(s // 2)), s) for s in image.shape)
#
#     # range for translation:
#     translation_range = tuple(s - cs for s, cs in zip(image.shape, cropped_shape))
#
#     # Instead of searching for all possible crops, we take into
#     # account the size of the crops to define a 'granularity' (
#     # stride) of the translations used for search:
#     granularity_factor = 4
#     granularity = tuple(cs // granularity_factor for cs in cropped_shape)
#
#     # grid for translations:
#     translation_indices = tuple(
#         max(1, int(granularity_factor * r / cs))
#         for r, cs in zip(translation_range, cropped_shape)
#     )
#
#     # Best scores and crop:
#     best_score = -1
#     best_slice = None
#     best_crop = None
#
#     for i, index in enumerate(numpy.ndindex(translation_indices)):
#         # translation:
#         translation = tuple(
#             int(i * cs / granularity_factor) for i, cs in
#             zip(index, cropped_shape)
#         )
#
#         # slice object for cropping:
#         crop_slice = tuple(
#             slice(t, t + s) for t, s in zip(translation, cropped_shape)
#         )
#
#         # extract crop:
#         crop = image[crop_slice]
#
#         score = evaluate_crop(
#             crop=crop,
#             mode=mode,
#             smoothing_size=smoothing_size,
#         )
#
#         if score > best_score:
#             best_score = score
#             best_slice = crop_slice
#             best_crop = crop
#
#
#     # If the best block is too small we abort and return the original image,
#     # because we are just 'pre-cropping' and we should still get a bigger crop:
#     if best_crop.size < crop_size:
#         return image, best_slice
#
#
#     # If the crop is fine then we proceed with the recursive call to further cutdown on the croip size:
#     precrop = _precrop(best_crop,
#                        crop_size,
#                        min_length,
#                        mode,
#                        smoothing_size)
#
#     # Return:
#     return precrop
#
