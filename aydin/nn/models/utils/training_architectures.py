import numpy

from aydin.nn.util.mask_generator import maskedgen, randmaskgen
from aydin.nn.util.validation_generator import val_data_generator


def get_jinet_fit_args(
    input_image,
    batch_size,
    total_num_patches,
    img_val,
    create_patches_for_validation,
    train_valid_ratio,
):
    """

    Parameters
    ----------
    input_image
    batch_size
    total_num_patches
    img_val
    create_patches_for_validation
    train_valid_ratio

    Returns
    -------
    validation_data

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
    patch_size=None,
    mask_size=None,
    val_marker=None,
    replace_by=None,
):
    """

    Parameters
    ----------
    train_method
    create_patches_for_validation
    input_image
    total_num_patches
    train_valid_ratio
    batch_size
    random_mask_ratio
    img_val
    patch_size
    mask_size
    val_marker
    replace_by

    Returns
    -------
    validation_data, validation_steps tuple

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
