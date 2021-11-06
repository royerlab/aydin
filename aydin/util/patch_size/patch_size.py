from typing import Tuple


def default_patch_size(image, patch_size, odd: bool = True) -> Tuple[int, ...]:
    """
    Returns  a normalised patch size given an image amd a parity.
    Returns
    -------
    object
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
