"""Validation data generation utilities for TensorFlow-based training.

Provides functions for creating training/validation image splits and
generating validation data batches with pixel-level masking.
"""

import math
from copy import deepcopy

import numpy
from scipy.ndimage.filters import convolve


def donut_filter(input_image):
    """Apply a donut-shaped averaging filter to the input image.

    Convolves the image with a kernel that averages neighbor pixels
    while excluding the center pixel. Used to generate replacement
    values for validation pixels.

    Parameters
    ----------
    input_image : numpy.ndarray
        Input image array with shape ``(B, ...spatial_dims..., C)``.

    Returns
    -------
    numpy.ndarray
        Filtered image with the same shape as the input.
    """
    k = [1] + [3 for _ in input_image.shape[1:-1]] + [1]
    kernel = numpy.ones(k) / 8
    k = [0] + [1 for _ in range(len(input_image.shape[1:-1]))] + [0]
    kernel[tuple(k)] = 0
    return convolve(input_image, kernel)


def train_image_generator(input_image, p=0.1):
    """Generate training and validation image sets with pixel-level masking.

    Randomly selects a fraction of pixels as validation pixels and
    replaces them in the training image with donut-filtered (neighbor
    averaged) values.

    Parameters
    ----------
    input_image : numpy.ndarray
        Input image array.
    p : float
        Fraction of pixels to use for validation.

    Returns
    -------
    img_train : numpy.ndarray
        Training image with validation pixels replaced by neighbor averages.
    input_image : numpy.ndarray
        Original image (used as validation reference).
    marker : numpy.ndarray
        Boolean mask marking validation pixel locations.
    """
    marker = numpy.random.uniform(size=input_image.shape)
    marker[marker > p] = 1
    marker[marker <= p] = 0
    marker = (1 - marker).astype(bool)

    img_filtered = donut_filter(input_image)
    img_train = deepcopy(input_image)
    img_train[marker] = img_filtered[marker]
    return img_train, input_image, marker


def val_data_generator(img_train, img_val, marker, batch_size, train_valid_ratio):
    """Create an infinite generator of validation data batches.

    Randomly samples a subset of images and yields batches of
    training inputs, mask markers, and validation targets.

    Parameters
    ----------
    img_train : numpy.ndarray
        Training image array.
    img_val : numpy.ndarray
        Validation reference image array.
    marker : numpy.ndarray
        Boolean mask marking validation pixels.
    batch_size : int
        Number of images per batch.
    train_valid_ratio : float
        Fraction of data used for validation.

    Yields
    ------
    tuple
        Tuple of (input_dict, target_batch) where input_dict contains
        ``'input'`` and ``'input_msk'`` keys.
    """
    val_ind = numpy.random.randint(
        0, img_train.shape[0], math.ceil(img_train.shape[0] * train_valid_ratio)
    )
    img_train = deepcopy(img_train[val_ind])
    img_val = deepcopy(img_val[val_ind])
    marker = deepcopy(marker[val_ind])

    j = 0
    num_cycle = numpy.ceil(img_train.shape[0] / batch_size)
    while True:
        i = numpy.mod(j, num_cycle).astype(int)
        train_batch = img_train[batch_size * i : batch_size * (i + 1)]
        val_batch = img_val[batch_size * i : batch_size * (i + 1)]
        marker_batch = marker[batch_size * i : batch_size * (i + 1)]
        j += 1
        yield {
            'input': train_batch,
            'input_msk': marker_batch.astype(numpy.float32),
        }, val_batch


# may be useful in the future for generating validation data with less memory used.
# def val_ind_generator(image_train, p=0.1):
#     """
#
#     Parameters
#     ----------
#     image_train
#         input image
#     p
#         ratio of pixels being used for validation
#
#     Returns
#     -------
#     training image
#         p pixels were replaced by surrounding average, val_pix_values: original image, mask: markers of validation pixels
#
#     """
#     marker = numpy.random.randint(0, image_train.size, size=int(image_train.size * p))
#     marker = numpy.unravel_index(marker, image_train.shape)
#     val_pix_values = image_train[marker]
#     img_filtered = donut_filter(image_train)
#     image_train[marker] = img_filtered[marker]
#     return image_train, val_pix_values, marker
