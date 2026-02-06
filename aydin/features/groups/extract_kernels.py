"""Utility functions for extracting representative kernels from images via clustering."""

from math import prod, sqrt
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import _extract_patches

from aydin.util.log.log import lprint, lsection


def extract_kernels(
    image: ArrayLike,
    size: int = 7,
    num_kernels: int = None,
    num_patches: int = 1e5,
    num_iterations: int = 100,
    display: bool = False,
) -> Sequence[ArrayLike]:
    """Extract representative kernels from an image using MiniBatchKMeans clustering.

    Extracts random patches from the image and clusters them to discover
    representative local patterns. The cluster centers are returned as
    normalized convolutional kernels.

    Parameters
    ----------
    image : ArrayLike
        Image to compute kernels for.
    size : int
        Size of each kernel along each dimension.
    num_kernels : int, optional
        Number of kernels to extract. If None, defaults to ``size ** ndim``.
    num_patches : int
        Total number of image patches to consider across all iterations.
    num_iterations : int
        Number of online learning iterations.
    display : bool
        When True, a matplotlib figure showing the learned kernels is
        displayed.

    Returns
    -------
    kernels : list of numpy.ndarray
        List of learned kernel arrays, each of shape ``(size,) * ndim``.
    """
    if num_kernels is None:
        num_kernels = size**image.ndim

    # #############################################################################
    # Learn the dictionary of images

    with lsection(
        f'Learning a dictionary of {num_kernels} kernels of size {size} from {num_patches} patches'
    ):
        rng = np.random.RandomState(0)
        kmeans = MiniBatchKMeans(n_clusters=num_kernels, random_state=rng, verbose=True)
        kernel_size = (size,) * image.ndim

        # The online learning part: cycle over the whole dataset 6 times
        for i in range(num_iterations):
            lprint(f"Iteration: {i}")
            data = extract_patches_nd(
                image, kernel_size, num_patches=int(num_patches) // num_iterations
            )
            data = np.reshape(data, (len(data), -1))
            data -= np.min(data, axis=0)
            data /= np.sum(data, axis=0)
            kmeans.partial_fit(data)

    if display:
        # #############################################################################
        # Plot the results
        plt.figure(figsize=(8, 8))
        for i, kernel in enumerate(kmeans.cluster_centers_):
            s = sqrt(num_kernels)
            plt.subplot(s + 1, s + 1, i + 1)
            plt.imshow(
                kernel.reshape(kernel_size),
                cmap=plt.cm.gray,
                interpolation='nearest',  # , vmin=0, vmax=1
            )
            plt.xticks(())
            plt.yticks(())

        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

        plt.show()

    return [np.reshape(kernel, kernel_size) for kernel in kmeans.cluster_centers_]


def extract_patches_nd(
    image, patch_shape: Union[int, Tuple[int, ...]], num_patches: Optional[int] = None
):
    """Extract patches from an n-dimensional image.

    The resulting patches are allocated in a dedicated array and can be
    randomly subsampled when ``num_patches`` is specified.

    Parameters
    ----------
    image : numpy.ndarray
        The original image data of arbitrary dimensionality.

    patch_shape : int or tuple of int
        The shape of one patch. Can be a single int (applied to all axes)
        or a tuple specifying the size along each axis.

    num_patches : int or float, optional
        The maximum number of patches to extract. If a float between 0 and 1,
        it is taken as a proportion of the total number of extractable patches.
        If None, all patches are returned.

    Returns
    -------
    patches : numpy.ndarray
        Array of shape ``(n_patches, *patch_shape)`` containing the extracted
        patches.
    """

    if type(patch_shape) is not tuple:
        patch_shape = (patch_shape,) * image.ndim

    extracted_patches = _extract_patches(
        image, patch_shape=patch_shape, extraction_step=1
    )

    extracted_patches_shape = extracted_patches.shape[: image.ndim]
    num_extracted_patches = prod(extracted_patches_shape)

    if num_patches is not None:
        if type(num_patches) is float:
            num_patches = int(num_patches * num_extracted_patches)

        if type(num_patches) is int:
            num_patches = min(num_patches, num_extracted_patches)

        p = num_patches / num_extracted_patches
        indices = np.random.choice(
            a=[False, True], size=extracted_patches_shape, p=[1 - p, p]
        )
        patches = extracted_patches[indices]

    else:
        patches = extracted_patches

    patches = patches.reshape(-1, *patch_shape)

    return patches
