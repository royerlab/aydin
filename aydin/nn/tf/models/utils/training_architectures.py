"""Training architecture configuration for TensorFlow models.

Provides helper functions to configure validation data and training
steps for different self-supervised training methods (shift-convolution,
checkerbox masking, random masking).
"""

import numpy

from aydin.nn.tf.util.mask_generator import maskedgen, randmaskgen
from aydin.nn.tf.util.validation_generator import val_data_generator


def get_jinet_fit_args(
    input_image,
    batch_size,
    total_num_patches,
    img_val,
    create_patches_for_validation,
    train_valid_ratio,
):
    """Configure validation data for JINet model training.

    Parameters
    ----------
    input_image : numpy.ndarray
        Training input image.
    batch_size : int
        Mini-batch size.
    total_num_patches : int
        Total number of training patches.
    img_val : numpy.ndarray or None
        Validation image (used when not creating patches for validation).
    create_patches_for_validation : bool
        If ``True``, splits patches for validation.
    train_valid_ratio : float
        Fraction of data used for validation.

    Returns
    -------
    tuple of numpy.ndarray
        Validation data as ``(input, target)`` tuple.
    """
    if create_patches_for_validation:
        val_split = total_num_patches * train_valid_ratio
        val_split = (
            val_split - (val_split % batch_size) + batch_size
        ) / total_num_patches
        validation_data = (
            input_image[int(total_num_patches * val_split) :],
            input_image[int(total_num_patches * val_split) :],
        )
    else:
        validation_data = (input_image, img_val)

    return validation_data


def get_unet_fit_args(
    train_method="random",
    create_patches_for_validation=False,
    input_image=None,
    total_num_patches=None,
    train_valid_ratio=None,
    batch_size=None,
    random_mask_ratio=None,
    img_val=None,
    mask_size=None,
    val_marker=None,
    replace_by=None,
):
    """Configure validation data and steps for UNet model training.

    Builds the appropriate validation data generator and computes
    the number of validation steps based on the chosen training method.

    Parameters
    ----------
    train_method : str
        Training method: ``'shiftconv'``, ``'checkerbox'``, ``'random'``,
        or ``'checkran'``.
    create_patches_for_validation : bool
        If ``True``, creates patches for validation.
    input_image : numpy.ndarray
        Training input image.
    total_num_patches : int
        Total number of training patches.
    train_valid_ratio : float
        Fraction of data used for validation.
    batch_size : int
        Mini-batch size.
    random_mask_ratio : float or None
        Fraction of pixels to mask in random masking mode.
    img_val : numpy.ndarray or None
        Validation image for pixel-level validation.
    mask_size : tuple of int or None
        Unit mask size for checkerbox masking.
    val_marker : numpy.ndarray or None
        Boolean mask for validation pixels.
    replace_by : str or None
        Replacement strategy for masked pixels.

    Returns
    -------
    validation_data : generator or tuple or None
        Validation data for the training loop.
    validation_steps : int or None
        Number of validation steps per epoch.
    """
    validation_steps = None
    validation_data = None

    if 'shiftconv' in train_method:
        if create_patches_for_validation:
            val_split = total_num_patches * train_valid_ratio
            val_split = (
                val_split - (val_split % batch_size) + batch_size
            ) / total_num_patches
            validation_data = (
                input_image[int(total_num_patches * val_split) :],
                input_image[int(total_num_patches * val_split) :],
            )
        else:
            validation_data = (input_image, img_val)
    elif 'checkerbox' in train_method:
        validation_steps = max(
            numpy.floor(input_image.shape[0] * train_valid_ratio / batch_size).astype(
                int
            ),
            1,
        )
        if create_patches_for_validation:
            validation_data = maskedgen(
                input_image, batch_size, mask_size, replace_by=replace_by
            )
        else:
            validation_data = val_data_generator(
                input_image,
                img_val,
                val_marker,
                batch_size,
                train_valid_ratio=train_valid_ratio,
            )
    elif 'random' in train_method:
        validation_steps = max(
            numpy.floor(total_num_patches * train_valid_ratio / batch_size).astype(int),
            1,
        )
        if create_patches_for_validation:
            validation_data = randmaskgen(
                input_image,
                batch_size,
                p_maskedpixels=random_mask_ratio,
                replace_by=replace_by,
            )
        else:
            validation_data = val_data_generator(
                input_image,
                img_val,
                val_marker,
                batch_size,
                train_valid_ratio=train_valid_ratio,
            )

    return validation_data, validation_steps
