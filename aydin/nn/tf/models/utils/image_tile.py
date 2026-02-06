"""Image tiling utilities for training data preparation.

Provides functions to split input and target images into training
and validation sets, with support for pixel-level validation masking.
"""

import numpy

from aydin.nn.tf.util.validation_generator import train_image_generator
from aydin.util.log.log import lsection


def tile_input_images(
    input_image,
    create_patches_for_validation,
    input_patch_idx,
    train_valid_ratio,
):
    """Split input images into training and validation sets.

    Creates training patches from the given indices and optionally
    generates pixel-level validation data by replacing validation
    pixels with neighbor averages.

    Parameters
    ----------
    input_image : numpy.ndarray
        Full input image array.
    create_patches_for_validation : bool
        If ``True``, creates patches for validation. If ``False``,
        uses pixel-level validation masking.
    input_patch_idx : list of int
        Indices of patches to use for training.
    train_valid_ratio : float
        Fraction of data used for validation.

    Returns
    -------
    img_train : numpy.ndarray
        Training image array.
    img_val : numpy.ndarray or None
        Validation image array (``None`` if patch-based validation).
    val_marker : numpy.ndarray or None
        Boolean mask marking validation pixels (``None`` if patch-based).
    """
    img_train_patch = []
    img_val = None
    val_marker = None

    if create_patches_for_validation:
        with lsection(
            f'Validation data will be created by monitoring {train_valid_ratio} of the patches/images in the input data.'
        ):
            for i in input_patch_idx:
                img_train_patch.append(input_image[i])
            img_train = numpy.vstack(img_train_patch)
    else:
        with lsection(
            f'Validation data will be created by monitoring {train_valid_ratio} of the pixels in the input data.'
        ):
            img_train, img_val, val_marker = train_image_generator(
                input_image, p=train_valid_ratio
            )

            img_val_patch = []
            marker_patch = []
            for i in input_patch_idx:
                img_train_patch.append(img_train[i])
                img_val_patch.append(img_val[i])
                marker_patch.append(val_marker[i])
            img_train = numpy.vstack(img_train_patch)
            img_val = numpy.vstack(img_val_patch)
            val_marker = numpy.vstack(marker_patch)

    return img_train, img_val, val_marker


def tile_target_images(
    img_train,
    target_image,
    input_patch_idx,
    self_supervised,
):
    """Prepare target images for training.

    For self-supervised training, uses the training image itself as the
    target. For supervised training, extracts target patches matching
    the input patch indices.

    Parameters
    ----------
    img_train : numpy.ndarray
        Training image array.
    target_image : numpy.ndarray
        Full target image array (ignored in self-supervised mode).
    input_patch_idx : list of int
        Indices of patches to extract from the target.
    self_supervised : bool
        If ``True``, returns ``img_train`` as the target.

    Returns
    -------
    numpy.ndarray
        Target image array for training.
    """
    if self_supervised:
        target_image = img_train
    else:
        target_patch = []
        for i in input_patch_idx:
            target_patch.append(target_image[i])
        target_image = numpy.vstack(target_patch)

    return target_image
