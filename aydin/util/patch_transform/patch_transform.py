from itertools import product
from math import prod
from typing import Tuple, Optional, Union
import numpy
from sklearn.feature_extraction.image import _extract_patches
from sklearn.utils import check_random_state

from aydin.util.array.outer import outer_sum


def extract_patches_nd(
    image,
    patch_size: Union[int, Tuple[int]] = 7,
    max_patches: Optional[int] = None,
    random_state=None,
):
    """Reshape a nD image into a collection of nD patches

    The resulting patches are allocated in a dedicated array.

    Does not support a channel dimensions, this has to be dealt with separately.

    Parameters
    ----------
    image : ndarray of shape (image_height, image_width) or \
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_size : tuple of int (patch_height, patch_width)
        The dimensions of one patch.

    max_patches : int or float, default=None
        The maximum number of patches to extract. If `max_patches` is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.

    random_state : int, RandomState instance, default=None
        Determines the random number generator used for random sampling when
        `max_patches` is not None. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    patches : array of shape (n_patches, patch_height, patch_width) or \
        (n_patches, patch_height, patch_width, n_channels)
        The collection of patches extracted from the image, where `n_patches`
        is either `max_patches` or the total number of patches that can be
        extracted.


    """

    # Normalises patch size to tuple:
    if type(patch_size) != tuple:
        patch_size = (patch_size,) * image.ndim

    # Check patch size versus image size:
    if any(s < ps for s, ps in zip(image.shape, patch_size)):
        raise ValueError(
            f"Patch size ({patch_size}) must be less than image size ({image.shape}) for all dimensions: "
        )

    # The next line requires contiguous arrays:
    image = numpy.ascontiguousarray(image)

    # O(1) patch extraction via numpy magic:
    extracted_patches = _extract_patches(
        image, patch_shape=patch_size, extraction_step=1
    )

    # Let's compute the number of patches:
    num_patches = prod(extracted_patches.shape[: image.ndim])

    if max_patches is not None:
        # Apply maximum:
        num_patches = min(num_patches, max_patches)

        # Check random state:
        rng = check_random_state(random_state)

        # Get indices:
        indices = tuple(
            rng.randint(s - ps + 1, size=num_patches)
            for s, ps in zip(image.shape, patch_size)
        )

        # Pull out selected patches:
        patches = extracted_patches[indices]
    else:
        # passthrough:
        patches = extracted_patches

    # Reshape:
    patches = patches.reshape((num_patches,) + patch_size)

    return patches


def reconstruct_from_nd_patches(
    patches, image_shape: Tuple[int], mode: str = 'windowed', gamma: float = 3
):
    """Reconstruct an nD image from all of its nD patches.

    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.

    Parameters
    ----------
    patches : ndarray of shape (n_patches, patch_height, patch_width) or \
        (n_patches, patch_height, patch_width, n_channels)
        The complete set of patches.

    image_shape : Tuple[int]
        The size of the image that will be reconstructed.

    mode: str
        Image reconstruction mode. Can be: 'mean' for computing the mean of all
        patches as they overlap to form the image, 'windowed' is the same but
        each patch voxel is weighted, and 'center' corresponds to only using the
        center pixel of each patch.

    gamma: float
        Parameter that controls how concentrated the patch weighting in the case
        of the 'windowed' reconstruction mode. When gamma tends to 0 then this
        becomes the same as the 'mean' reconstruction mode, if gamma is above 1
        then the weight function is concentrated around the center of the patch
        and becomes simlar to 'center' mode as gamma tends to infinity.


    Returns
    -------
    image : ndarray of shape image_size
        The reconstructed image.
    """

    # Patch size:
    patch_shape = patches.shape[1:]

    # We allocate the reconstructed image:
    image = numpy.zeros(image_shape, dtype=numpy.float32)

    if mode == 'mean' or mode == 'windowed':

        # First we compute the indices over which to scan:
        ranges = tuple(range(s - ps + 1) for s, ps in zip(image_shape, patch_shape))

        # Weight counts:
        weights = numpy.zeros(image_shape, dtype=numpy.float32)

        # precompute window:
        window = _window(patch_shape, gamma)

        if mode == 'windowed':
            # Apply window to patches:
            patches = patches * window[numpy.newaxis, ...]

        # Reconstruct image:
        for p, index in zip(patches, product(*ranges)):
            slicing = tuple(slice(i, i + ps) for i, ps in zip(index, patch_shape))
            image[slicing] += p
            weights[slicing] += window

        # Divide by weights sum::
        image /= weights

        # Correct issues with weights:
        image[weights == 0] = 0

    elif mode == 'center':
        # First we compute the indices over which to scan:
        ranges = tuple(
            range(ps // 2, s - (ps // 2 + 1) + 1)
            for s, ps in zip(image_shape, patch_shape)
        )

        slicing = tuple(slice(ps // 2, (ps // 2) + 1) for ps in patch_shape)
        for p, index in zip(patches, product(*ranges)):
            image[index] += p[slicing]

    # Convert image back to original dtype:
    image = image.astype(dtype=patches.dtype, copy=False)

    return image


def _window(shape: Tuple[int], gamma: float):
    if gamma < 0.01:
        # gamma->0 => uniform window!
        window_nd = numpy.ones(shape)
    else:
        window_tuple = (numpy.abs(numpy.blackman(s)) for s in shape)
        window_nd = numpy.sqrt(outer_sum(*window_tuple)) + 1e-6
        window_nd **= gamma
    window_nd = numpy.ascontiguousarray(window_nd)
    window_nd /= numpy.sum(window_nd)
    return window_nd
