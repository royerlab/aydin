from aydin.util.log.log import lprint


def get_ideal_patch_size(nb_unet_levels, training_architecture):
    patch_size = (
        get_receptive_field_radius(
            nb_unet_levels,
            shiftconv='shiftconv' == training_architecture,
        )
        * 2
    )

    patch_size -= patch_size % 2**nb_unet_levels

    if patch_size < 2**nb_unet_levels:
        raise ValueError('Number of layers is too large for given patch size.')

    lprint(f'Patch size: {patch_size}')
    return patch_size


def get_receptive_field_radius(nb_unet_levels: int, shiftconv: bool = False) -> int:
    """Returns the radius of the anticipated receptive
    field of the UNet of interest.

    Parameters
    ----------
    nb_unet_levels : int
    shiftconv : bool

    Returns
    -------
    int

    """
    if shiftconv:
        rf = 7 if nb_unet_levels == 0 else 36 * 2 ** (nb_unet_levels - 1) - 6
    else:
        rf = 3 if nb_unet_levels == 0 else 18 * 2 ** (nb_unet_levels - 1) - 4
    return int(rf // 2)
