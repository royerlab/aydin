import math
from typing import Optional, Union, Tuple
import numpy
from pynndescent import NNDescent
from scipy.fft import dctn, idctn

from aydin.util.patch_size.patch_size import default_patch_size
from aydin.util.patch_transform.patch_transform import (
    extract_patches_nd,
    reconstruct_from_nd_patches,
)


def calibrate_denoise_bmnd(
    image, patch_size: Optional[Union[int, Tuple[int], str]] = None
):
    """
    Calibrates the BMnD denoiser for the given image and
    returns the optimal parameters obtained using the N2S
    loss.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate spectral denoiser for.

    patch_size: int
        Patch size for the 'image-to-patch' transform.
        Can be an int s that corresponds to isotropic
        patches of shape: (s,)*image.ndim, or a tuple
        of ints. By default (None) the patch size is
        chosen automatically to give the best results.


    Returns
    -------
    Denoising function, dictionary containing optimal parameters,
    and free memory needed in bytes for computation.
    """

    # Default patch sizes vary with image dimension:
    patch_size = default_patch_size(image, patch_size, odd=True)

    best_parameters = {}

    # Memory needed:
    memory_needed = 2 * image.nbytes + 8 * image.nbytes * math.prod(patch_size)

    return denoise_bmnd, best_parameters, memory_needed


def denoise_bmnd(
    image,
    patch_size: Optional[Union[int, Tuple[int]]] = None,
    block_depth: Optional[int] = None,
    mode: str = 'median',
    reconstruction_gamma: float = 0,
    multi_core: bool = True,
):
    """
    Denoises the given image using the Block-Matching-nD approach.
    Our implementation follows the general outline of <a
    href="https://en.wikipedia.org/wiki/Block-matching_and_3D_filtering">
    Block-matching and 3D filtering (BM3D)</a> approaches.
    \n\n
    Note: This is currently only usable for small images, even for
    moderately sized images the computation time is prohibitive. We
    are planning to implement a GPU version to make it more usefull.


    Parameters
    ----------
    image: ArrayLike
        Image to denoise

    patch_size: int
        Patch size for the 'image-to-patch' transform.
        Can be: 'full' for a single patch covering the whole image, 'half', 'quarter',
        or an int s that corresponds to isotropic patches of shape: (s,)*image.ndim,
        or a tuple of ints. By default (None) the patch size is chosen automatically
        to give the best results.

    block_depth: int or None for default
        Block depth.

    mode: str
        Possible modes are: 'median', 'mean'.

    threshold: float
        Threshold between 0 and 1

    freq_bias_stength: float
        Frequency bias: closer to zero: no bias against high frequencies,
        closer to one and above: stronger bias towards high-frequencies.

    freq_cutoff: float
        Cutoff frequency, must be within [0, 1]. In addition

    order: float
        Filter order, typically an integer above 1.

    reconstruction_gamma: float
        Patch reconstruction parameter

    multi_core: bool
        By default we use as many cores as possible, in some cases, for small
        (test) images, it might be faster to run on a single core instead of
        starting the whole parallelization machinery.


    Returns
    -------
    Denoised image

    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # Normalise patch size:
    patch_size = default_patch_size(image, patch_size, odd=True)

    # Default value for block_depth:
    if block_depth is None:
        if type(patch_size) is int:
            block_depth = patch_size
        else:
            block_depth = max(patch_size)

    # First we apply the patch transform:
    patches = extract_patches_nd(image, patch_size=patch_size)

    axes = tuple(1 + a for a in range(image.ndim))
    patches = dctn(patches, axes=axes, workers=-1 if multi_core else 1)

    # reshape patches as vectors:
    original_patches_shape = patches.shape
    patches = patches.reshape(patches.shape[0], -1)

    # Build the data structure for nearest neighbor search:
    nn = NNDescent(patches, n_jobs=-1 if multi_core else 1)

    # for each patch we query for the nearest patch:
    indices, distances = nn.query(patches, k=block_depth)

    # reshape patches back to their original shape:
    patches = patches.reshape(original_patches_shape)

    # Extract blocks:
    blocks = patches[indices]

    # Denoise blocks:
    if mode == 'median':
        patches = numpy.median(blocks, axis=1)
    elif mode == 'mean':
        patches = numpy.mean(blocks, axis=1)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    del blocks

    patches = idctn(patches, axes=axes, workers=-1 if multi_core else 1)

    # Transform back from patches to image:
    denoised_image = reconstruct_from_nd_patches(
        patches, image.shape, gamma=reconstruction_gamma
    )

    # Cast back to float32 if needed:
    denoised_image = denoised_image.astype(numpy.float32, copy=False)

    return denoised_image
