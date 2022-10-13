import math

import numpy
from copy import deepcopy
from scipy.ndimage.filters import convolve


def donut_filter(input_image):
    """
    Parameters
    ----------
    input_image

    Returns
    -------
    convolved image with kernel
    """
    k = [1] + [3 for _ in input_image.shape[1:-1]] + [1]
    kernel = numpy.ones(k) / 8
    k = [0] + [1 for _ in range(len(input_image.shape[1:-1]))] + [0]
    kernel[tuple(k)] = 0
    return convolve(input_image, kernel)


def train_image_generator(input_image, p=0.1):
    """
    Parameters
    ----------
    input_image
        input image
    p
        ratio of pixels being used for validation

    Returns
    -------
    training image
        p pixels were repalced by surrounding average, validation image: original image, mask: markers of validation
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
    """
    Parameters
    ----------
    img_train
    img_val
    marker
    batch_size
    train_valid_ratio
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
#         p pixels were repalced by surrounding average, val_pix_values: original image, mask: markers of validation pixels
#
#     """
#     marker = numpy.random.randint(0, image_train.size, size=int(image_train.size * p))
#     marker = numpy.unravel_index(marker, image_train.shape)
#     val_pix_values = image_train[marker]
#     img_filtered = donut_filter(image_train)
#     image_train[marker] = img_filtered[marker]
#     return image_train, val_pix_values, marker
