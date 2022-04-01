import numpy


def calc_patch_size(
    nb_unet_levels,
    model_architecture,
    training_architecture,
    spacetime_ndim,
):
    is_shiftconv = 'shiftconv' == training_architecture

    patch_size = unet_receptive_field_radius(nb_unet_levels, shiftconv=is_shiftconv) * 2

    patch_size -= (patch_size % (2 ** nb_unet_levels))

    if patch_size < 2 ** nb_unet_levels:
        raise ValueError(
            'Number of layers is too large for given patch size.'
        )

    # Adjust patch_size for given input shape
    patch_size = [patch_size] * spacetime_ndim

    # Check patch_size for unet models
    if 'unet' in model_architecture:
        patch_size = numpy.array(patch_size)
        if (patch_size.max() / (2 ** nb_unet_levels) <= 0).any():
            raise ValueError(
                f'Tile size is too small. The largest dimension of tile size has to be >= {2 ** nb_unet_levels}.'
            )
        if (patch_size[-2:] % 2 ** nb_unet_levels != 0).any():
            raise ValueError(
                f'Tile sizes on XY plane have to be multiple of 2^{nb_unet_levels}'
            )

    # Check if the smallest dimension of input data >= patch_size
    if min(patch_size) > min(input_dim[:-1]):
        smallest_dim = min(input_dim[:-1])
        patch_size[numpy.argsort(input_dim[:-1])[0]] = (
                smallest_dim // 2 * 2
        )

    return patch_size


def unet_receptive_field_radius(
    nb_unet_levels,
    shiftconv,
):
    if shiftconv:
        rf = 7 if nb_unet_levels == 0 else 36 * 2 ** (nb_unet_levels - 1) - 6
    else:
        rf = 3 if nb_unet_levels == 0 else 18 * 2 ** (nb_unet_levels - 1) - 4
    return int(rf // 2)


# def calc_nb_patches(
#
# ):
#     # Determine total number of patches
#     if total_num_patches is None:
#         total_num_patches = min(
#             input_image.size / numpy.prod(patch_size), 10240
#         )  # upper limit of num of patches
#         total_num_patches = (
#                 total_num_patches
#                 - (total_num_patches % batch_size)
#                 + batch_size
#         )
#     else:
#         if total_num_patches < batch_size:
#             raise ValueError(
#                 'total_num_patches has to be larger than batch_size.'
#             )
#         total_num_patches = (
#                 total_num_patches
#                 - (total_num_patches % batch_size)
#                 + batch_size
