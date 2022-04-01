import numpy

from aydin.nn.util.data_util import random_sample_patches
from aydin.util.log.log import lsection, lprint


def patch_generation(
        input_image,
        target_image,
        patch_size,
        nb_patches,
        adoption_rate,
        _create_patches_for_validation,
        self_supervised,
):
    with lsection('Random patch sampling...'):
        lprint(f'Total number of patches: {nb_patches}')
        input_patch_idx = random_sample_patches(
            input_image,
            patch_size,
            nb_patches,
            adoption_rate,
        )

        nb_patches = len(input_patch_idx)

        img_train_patch = []

        if _create_patches_for_validation:
            for i in input_patch_idx:
                img_train_patch.append(input_image[i])
            img_train = numpy.vstack(img_train_patch)
        else:
            img_val_patch = []
            marker_patch = []
            for i in input_patch_idx:
                img_train_patch.append(img_train[i])
                img_val_patch.append(img_val[i])
                marker_patch.append(val_marker[i])
            img_train = numpy.vstack(img_train_patch)
            img_val = numpy.vstack(img_val_patch)
            val_marker = numpy.vstack(marker_patch)
            validation_images = img_val
            validation_markers = val_marker

        if not self_supervised:
            target_patch = []
            for i in input_patch_idx:
                target_patch.append(target_image[i])
            target_image = numpy.vstack(target_patch)
        else:
            target_image = img_train

        return input_image, target_image, nb_patches
