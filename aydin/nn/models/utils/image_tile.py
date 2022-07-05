import numpy

from aydin.nn.util.data_util import random_sample_patches
from aydin.util.log.log import lsection, lprint


def tile_input_and_target_images(
    input_image,
    target_image,
    img_train,
    img_val,
    val_marker,
    patch_size: int,
    total_num_patches: int,
    adoption_rate,
    create_patches_for_validation,
    self_supervised,
):
    # Tile input and target image
    if patch_size is not None:
        with lsection('Random patch sampling...'):
            lprint(f'Total number of patches: {total_num_patches}')
            input_patch_idx = random_sample_patches(
                input_image,
                patch_size,
                total_num_patches,
                adoption_rate,
            )

            total_num_patches = len(input_patch_idx)

            img_train_patch = []

            if create_patches_for_validation:
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
                # validation_images = img_val
                # validation_markers = val_marker

            if not self_supervised:
                target_patch = []
                for i in input_patch_idx:
                    target_patch.append(target_image[i])
                target_image = numpy.vstack(target_patch)
            else:
                target_image = img_train

        # TODO: return
