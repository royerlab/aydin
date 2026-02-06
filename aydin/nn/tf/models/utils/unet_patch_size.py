"""Patch size calculation and validation for UNet models.

Provides functions to compute the ideal patch size based on the UNet
receptive field, calculate theoretical receptive field sizes, and
validate input dimensions for shift-convolution architectures.
"""

import numpy

from aydin.util.log.log import lprint


def get_ideal_patch_size(nb_unet_levels, training_architecture):
    """Compute the ideal patch size for a UNet model.

    Calculates a patch size that is twice the receptive field radius
    and aligned to the UNet level stride.

    Parameters
    ----------
    nb_unet_levels : int
        Number of UNet encoder/decoder levels.
    training_architecture : str
        Training architecture type (e.g., ``'shiftconv'``).

    Returns
    -------
    int
        Ideal patch size.

    Raises
    ------
    ValueError
        If the computed patch size is too small for the number of levels.
    """
    patch_size = (
        unet_receptive_field_radius(
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


def unet_receptive_field_radius(nb_unet_levels: int, shiftconv: bool = False) -> int:
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


def post_tiling_patch_size_validation(
    img_train,
    nb_unet_levels,
    training_architecture,
    self_supervised,
):
    """
    Last check of input size especially for shiftconv

    Parameters
    ----------
    img_train
    nb_unet_levels
    training_architecture
    self_supervised

    """
    if 'shiftconv' == training_architecture and self_supervised:
        if (
            numpy.mod(
                img_train.shape[1:][:-1],
                numpy.repeat(2**nb_unet_levels, len(img_train.shape[1:][:-1])),
            )
            != 0
        ).any():
            raise ValueError(
                'Each dimension of the input image has to be a multiple of 2^nb_unet_levels for shiftconv.'
            )
        lprint(
            'Model will be generated for self-supervised learning with shift convolution scheme.'
        )
        if numpy.diff(img_train.shape[1:][:2]) != 0:
            raise ValueError(
                'Make sure the input image shape is cubic as shiftconv mode involves rotation.'
            )
        if (
            numpy.mod(
                img_train.shape[1:][:-1],
                numpy.repeat(
                    2 ** (nb_unet_levels - 1),
                    len(img_train.shape[1:][:-1]),
                ),
            )
            != 0
        ).any():
            raise ValueError(
                'Each dimension of the input image has to be a multiple of '
                '2^(nb_unet_levels-1) as shiftconv mode involves pixel shift. '
            )
