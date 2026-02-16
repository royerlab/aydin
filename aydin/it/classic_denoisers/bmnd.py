"""Block-Matching nD (BMnD) denoiser.

Provides calibration and denoising functions using the Block-Matching nD
approach, a generalization of BM3D. Patches are extracted, transformed
to the DCT domain, grouped by nearest-neighbor search, and aggregated
(via median or mean) to produce denoised patches.
"""

import math
from typing import Optional, Tuple, Union

import numpy
from numpy.typing import ArrayLike
from pynndescent import NNDescent
from scipy.fft import dctn, idctn

from aydin.util.log.log import aprint, asection
from aydin.util.patch_size.patch_size import default_patch_size
from aydin.util.patch_transform.patch_transform import (
    extract_patches_nd,
    reconstruct_from_nd_patches,
)


def calibrate_denoise_bmnd(
    image: ArrayLike, patch_size: Optional[Union[int, Tuple[int], str]] = None
):
    """
    Calibrates the BMnD denoiser for the given image and
    returns the optimal parameters obtained using the N2S
    loss.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate spectral denoiser for.

    patch_size : int, tuple of int, or None
        Patch size for the 'image-to-patch' transform.
        Can be an int s that corresponds to isotropic
        patches of shape: (s,)*image.ndim, or a tuple
        of ints. By default (None) the patch size is
        chosen automatically to give the best results.

    Returns
    -------
    denoise_function : callable
        The ``denoise_bmnd`` function.
    best_parameters : dict
        Dictionary of optimal denoising parameters.
    memory_needed : int
        Estimated memory needed in bytes for denoising the full image.
    """

    # Default patch sizes vary with image dimension:
    patch_size = default_patch_size(image, patch_size, odd=True)
    aprint(f"BMnD calibration: patch_size={patch_size}")

    best_parameters = {}

    # Memory needed:
    memory_needed = 2 * image.nbytes + 8 * image.nbytes * math.prod(patch_size)

    return denoise_bmnd, best_parameters, memory_needed


def denoise_bmnd(
    image: ArrayLike,
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
    are planning to implement a GPU version to make it more useful.
    <notgui>

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
        Block depth, i.e. number of nearest-neighbor patches to aggregate.
        If None, defaults to the maximum of the patch size dimensions.

    mode: str
        Aggregation mode for matched blocks. Possible modes are:
        'median' and 'mean'.

    reconstruction_gamma: float
        Patch reconstruction parameter that controls blending of
        overlapping patches. A value of 0 gives uniform weighting.

    multi_core : bool
        By default we use as many cores as possible, in some cases, for small
        (test) images, it might be faster to run on a single core instead of
        starting the whole parallelization machinery.

    Returns
    -------
    numpy.ndarray
        Denoised image as a float32 array.

    Raises
    ------
    ValueError
        If ``mode`` is not 'median' or 'mean'.
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

    with asection(
        f"Denoise image of shape {image.shape} with BMnD (patch_size={patch_size}, block_depth={block_depth})"
    ):
        with asection("Extract patches and DCT"):
            patches = extract_patches_nd(image, patch_size=patch_size)
            axes = tuple(1 + a for a in range(image.ndim))
            patches = dctn(patches, axes=axes, workers=-1 if multi_core else 1)

        with asection("Nearest-neighbor search"):
            original_patches_shape = patches.shape
            patches = patches.reshape(patches.shape[0], -1)
            nn = NNDescent(patches, n_jobs=-1 if multi_core else 1)
            indices, distances = nn.query(patches, k=block_depth)
            patches = patches.reshape(original_patches_shape)

        with asection(f"Aggregate matched blocks ({mode})"):
            blocks = patches[indices]
            if mode == 'median':
                patches = numpy.median(blocks, axis=1)
            elif mode == 'mean':
                patches = numpy.mean(blocks, axis=1)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            del blocks

        with asection("Inverse DCT and reconstruct"):
            patches = idctn(patches, axes=axes, workers=-1 if multi_core else 1)
            denoised_image = reconstruct_from_nd_patches(
                patches, image.shape, gamma=reconstruction_gamma
            )
            denoised_image = denoised_image.astype(numpy.float32, copy=False)

    return denoised_image
