import math
import time
from random import randrange
from typing import Optional

import numpy
from scipy.ndimage import sobel, laplace

from aydin.util.fast_uniform_filter.uniform_filter import uniform_filter_auto
from aydin.util.log.log import lprint


def representative_crop(
    image,
    mode: str = 'sobelmin',
    crop_size: Optional[int] = None,
    min_length: int = 16,
    smoothing_size: int = 1.0,
    equal_sides: bool = False,
    favour_odd_lengths: bool = False,
    fast_mode: bool = False,
    fast_mode_num_crops: int = 1500,
    max_time_in_seconds: float = 1,
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

    smoothing_size : int
        Uniform filter smoothing to achieve some crude denoising and thus
        make it a bit easier to estimate the score.

    equal_sides : bool
        When True the crop will have all its sides of equal size (square, cube, ...)

    favour_odd_lengths : bool
        If possible favours crops that have odd shape lengths.

    fast_mode: bool
        In fast mode we pick random crops instead of checking every possible strided crop.

    fast_mode_num_crops: int
        Number of crops to check in fast mode.

    max_time_in_seconds: float
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

    # Number of voxels in image:
    num_voxels = image.size

    # Default number of voxels:
    if crop_size is None:
        crop_size = 32000

    # Ratio by which to crop to acheive max num voxels:
    ratio = (crop_size / num_voxels) ** (1 / image.ndim)

    if ratio >= 1:
        # If the image is small enough no point in getting a crop!
        return image

    # cropped shape:
    cropped_shape = tuple(min(max(min_length, int(s * ratio)), s) for s in image.shape)

    # Favour odd lengths if requested:
    if favour_odd_lengths:
        cropped_shape = tuple((s // 2) * 2 + 1 for s in cropped_shape)

    # We enforce equal sides if requested:
    if equal_sides:
        min_length = min(cropped_shape)
        cropped_shape = tuple((min_length,) * image.ndim)

    # range for translation:
    translation_range = tuple(s - cs for s, cs in zip(image.shape, cropped_shape))

    # grid for translations:
    translation_indices = tuple(
        max(1, int(2 * r / cs)) for r, cs in zip(translation_range, cropped_shape)
    )

    # min and max for crop value normalisation:
    image_min = None
    image_max = None

    # We loop through a number of crops and keep the one wit the best score:
    best_score = -1
    best_slice = None
    best_crop = None

    if fast_mode and image.size > 1e6:

        # We make sure that the number of crops is not too large given
        # the relative size of the crop versus whole image:
        fast_mode_num_crops = min(fast_mode_num_crops, int(image.size / crop_size))

        for index in range(fast_mode_num_crops):

            # translation:
            translation = tuple(
                (randrange(0, s - cs) if cs != s else 0)
                for s, cs in zip(image.shape, cropped_shape)
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
            crop = image[crop_slice]

            score = evaluate_crop(
                crop=crop,
                image_min=image_min,
                image_max=image_max,
                mode=mode,
                smoothing_size=smoothing_size,
            )

            # slice object for the actual crop:
            crop_slice = _crop_slice(translation, cropped_shape, 1)

            # update best score and image:
            if score > best_score and not math.isinf(score):
                best_score = score
                best_slice = crop_slice

                # We make sure to have the full and original crop!
                best_crop = image[best_slice]

            if time.time() > start_time + max_time_in_seconds:
                lprint("Interrupting crop search because of timeout!")
                break

    else:
        for index in numpy.ndindex(translation_indices):

            # translation:
            translation = tuple(int(i * cs / 2) for i, cs in zip(index, cropped_shape))

            # slice object for cropping:
            crop_slice = tuple(
                slice(t, t + s) for t, s in zip(translation, cropped_shape)
            )

            # extract crop:
            crop = image[crop_slice]

            score = evaluate_crop(
                crop=crop,
                image_min=image_min,
                image_max=image_max,
                mode=mode,
                smoothing_size=smoothing_size,
            )

            # update best score and image:
            if score > best_score and not math.isinf(score):
                best_score = score
                best_slice = crop_slice

                # We make sure to have the full and original crop!
                best_crop = image[best_slice]

            if time.time() > start_time + max_time_in_seconds:
                lprint("Interrupting crop search because of timeout!")
                break

    if display_crop:
        smoothed_best_crop = _smoothing(best_crop, size=smoothing_size)

        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name='image')
            viewer.add_image(best_crop, name='best_crop')
            viewer.add_image(smoothed_best_crop, name='smoothed_best_crop')

    if return_slice:
        return best_crop, best_slice
    else:
        return best_crop


def evaluate_crop(crop, image_min, image_max, mode, smoothing_size):
    # smooth crop as a crude denoising:
    smoothed_crop = _smoothing(crop, size=smoothing_size)
    # convert type:
    smoothed_crop = smoothed_crop.astype(dtype=numpy.float32, copy=False)

    if image_min is None or image_max is None:
        image_min = crop.min()
        image_max = crop.max()
    else:
        # Normalise crop values:
        smoothed_crop -= image_min
        smoothed_crop /= image_max
    # compute score:
    if mode == 'contrast':
        score = numpy.std(smoothed_crop)
    elif mode == 'sobel':
        score = numpy.std(_sobel_magnitude(smoothed_crop))
    elif mode == 'sobelmin':
        score = numpy.std(_sobel_minimum(smoothed_crop))
    elif mode == 'laplace':
        score = numpy.std(laplace(smoothed_crop))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return score


def _smoothing(crop, size):
    smoothed_crop = uniform_filter_auto(crop, size=1 + 2 * size, printout_choice=False)
    # smoothed_crop = gaussian_filter(crop, sigma=size)
    return smoothed_crop


def _sobel_magnitude(image):

    magnitude = numpy.zeros_like(image)
    for axis in range(image.ndim):
        magnitude += sobel(image, axis=axis) ** 2

    return numpy.sqrt(magnitude)


def _sobel_minimum(image):

    minimum = None
    for axis in range(image.ndim):
        if minimum is None:
            minimum = sobel(image, axis=axis)
        else:
            minimum = numpy.minimum(minimum, sobel(image, axis=axis))

    return minimum
