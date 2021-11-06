from copy import deepcopy
import numpy as np
from scipy.ndimage import median_filter

from aydin.util.log.log import lprint


def masker(batch_vol, i=None, mask_shape=None, p=None):
    """
    Single mask generator.

    Parameters
    ----------
    batch_vol
        batch volume, desn't include batch and ch dimensions
    i
    mask_shape
        mask shape e.g. (3, 3)
    p
        possibility of masked pixels on random masking approach

    Returns
    -------
    mask

    Notes
    -----
    Spec.
    The core function to create mask fixed_pattern.
    Can create both checkerbox and random masks.
    The output should have the same dimension size as the that of the image to be masked.

    Input
    patch size:
    iteration index: generate different mask patterns when iteration index changes
    mask size: a unit size of mask. e.g. unit size of checkerbox
    mask ratio for random masking: how much % of the pixels should be masked in random masking method
    """

    if p:
        mask = np.random.uniform(size=batch_vol)
        mask[mask > p] = 1
        mask[mask <= p] = 0
        mask = mask.astype(bool)
    else:
        i = i % np.prod(mask_shape)
        mask = np.zeros(np.prod(mask_shape), dtype=bool)
        mask[i] = True
        mask = mask.reshape(mask_shape)
        rep = np.ceil(np.asarray(batch_vol) / np.asarray(mask_shape)).astype(int)
        mask = np.tile(mask, tuple(rep))
        ind = tuple([slice(batch_vol[i]) for i in range(len(batch_vol))])
        mask = mask[ind]
    return mask


def med_filter(input_image):
    """
    Parameters
    ----------
    input_image

    Returns
    -------
    filtered image

    Notes
    -----
    Spec.
    This median filter function is used to replace validation pixels the median values of surround pixels.
    """
    k = [1] + [3 for _ in input_image.shape[1:-1]] + [1]
    kernel = np.ones(k)
    k = [0] + [1 for _ in range(len(input_image.shape[1:-1]))] + [0]
    kernel[tuple(k)] = 0
    return median_filter(input_image, footprint=kernel)


def maskedgen(image, batch_size, mask_size, replace_by='zero'):
    """
    Mask generator. Returns a generator.

    Parameters
    ----------
    image
        the image to be masked
    batch_size
        generator will generate <mini batch size> of slices a time to feed to CNN model
    mask_size
        a unit size of checkerbox
    replace_by
        an argument to select whether replacing masked pixels with zero or median values of the surrounding pixels

    Returns
    -------
    mask generator
    """
    patch_size = image.shape[1:-1]
    img_output = deepcopy(image)

    j = 0
    num_cycle = np.ceil(img_output.shape[0] / batch_size)
    lprint(f'Masked pixels are replaced by {replace_by}')
    while True:
        i = np.mod(j, num_cycle).astype(int, copy=False)
        image_batch = img_output[batch_size * i : batch_size * (i + 1)]
        for i in range(int(np.prod(mask_size))):
            mask = masker(
                patch_size, i, mask_size, p=None
            )  # generate a 2D mask with same size of image_batch; True value is masked.
            masknega = np.broadcast_to(
                np.expand_dims(np.expand_dims(mask, 0), -1), image_batch.shape
            )  # broadcast the 2D mask to batch dimension ready for multiplication
            train_img = (
                np.broadcast_to(
                    np.expand_dims(np.expand_dims(~mask, 0), -1), image_batch.shape
                )
                * image_batch
            )  # pixels in training image are blocked by multiply by 0

            if replace_by == 'random':
                train_img = train_img + np.random.random(patch_size) * masknega
            elif replace_by == 'median':
                train_img = train_img + med_filter(image_batch) * masknega

            target_img = masknega * image_batch
            j += 1
            yield {
                'input': train_img,
                'input_msk': masknega.astype(np.float32),
            }, target_img


def randmaskgen(
    image,
    batch_size,
    p_maskedpixels=None,
    replace_by='zero',  # 'zero' or 'random' or 'median'
):
    """
    Create a random mask generator.

    Parameters
    ----------
    image
        image to be masked
    batch_size
        generator will generate <mini batch size> of slices a time to feed to CNN model
    p_maskedpixels
        how much % of the pixels should be masked in random masking method
    replace_by
        an argument to select whether replacing masked pixels with zero or median values of the surrounding pixels

    Returns
    -------
    random mask generator

    """
    batch_vol = (batch_size,) + image.shape[1:]
    img_output = deepcopy(image)

    j = 0
    num_cycle = np.ceil(img_output.shape[0] / batch_size)
    lprint(f'Masked pixels are replaced by {replace_by}')
    while True:
        i = np.mod(j, num_cycle).astype(int, copy=False)
        image_batch = img_output[batch_size * i : batch_size * (i + 1)]

        mask = masker(
            batch_vol, p=p_maskedpixels
        )  # generate a 2D mask with same size of image_batch; p of the pix are 0
        train_img = (
            mask * image_batch
        )  # pixels in training image are blocked by multiply by 0
        masknega = ~mask  # p of the pixels are 1

        if replace_by == 'random':
            train_img = train_img + np.random.random(batch_vol) * masknega
        elif replace_by == 'median':
            train_img = train_img + med_filter(image_batch) * masknega

        target_img = masknega * image_batch
        j += 1

        yield {'input': train_img, 'input_msk': masknega.astype(np.float32)}, target_img
