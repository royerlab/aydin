import numpy

from aydin.nn.util.validation_generator import train_image_generator
from aydin.util.log.log import lsection


def tile_input_images(
    input_image,
    create_patches_for_validation,
    input_patch_idx,
    train_valid_ratio,
):
    img_train_patch = []

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
    if self_supervised:
        target_image = img_train
    else:
        target_patch = []
        for i in input_patch_idx:
            target_patch.append(target_image[i])
        target_image = numpy.vstack(target_patch)

    return target_image
