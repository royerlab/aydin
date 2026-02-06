"""Default patch size computation for image denoising.

Provides a utility function to determine appropriate patch sizes based
on image dimensionality and shape.
"""

from typing import Tuple


def default_patch_size(image, patch_size, odd: bool = True) -> Tuple[int, ...]:
    """Compute a normalized patch size for a given image.

    Selects a default patch size based on image dimensionality if none
    is provided, normalizes scalar sizes to tuples, and ensures the
    patch size does not exceed half the image dimensions.

    Parameters
    ----------
    image : numpy.ndarray
        Image for which to determine the patch size.
    patch_size : int, tuple of int, or None
        Requested patch size. If None, a default is chosen based on
        the number of image dimensions.
    odd : bool
        If True and ``patch_size`` is None, prefer odd default sizes;
        otherwise prefer even sizes.

    Returns
    -------
    tuple of int
        Normalized patch size tuple with one entry per image dimension.
    """
    # Default patch sizes vary with image dimension:
    if patch_size is None:
        if image.ndim == 1:
            patch_size = 17 if odd else 16
        elif image.ndim == 2:
            patch_size = 8 if odd else 7
        elif image.ndim == 3:
            patch_size = 6 if odd else 5
        elif image.ndim == 4:
            patch_size = 4 if odd else 3

    # Normalise to tuple:
    if type(patch_size) is not tuple:
        patch_size = (patch_size,) * image.ndim

    # Ensure patch size is not larger than image:
    patch_size = tuple(
        min(ps, max(3, s // 2)) for ps, s in zip(patch_size, image.shape)
    )

    return patch_size
