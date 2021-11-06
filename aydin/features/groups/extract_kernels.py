from functools import reduce
from math import sqrt
from operator import mul
from typing import Tuple, Union, Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import _extract_patches

from aydin.util.log.log import lprint, lsection


def extract_kernels(
    image,
    size: int = 7,
    num_kernels: int = None,
    num_patches: int = 1e5,
    num_iterations: int = 100,
    display: bool = False,
):
    if num_kernels is None:
        num_kernels = size ** image.ndim

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
    """Reshape a 2D image into a collection of patches

    The resulting patches are allocated in a dedicated array.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    image : ndarray of shape (image_height, image_width) or \
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_shape : int or tuple of ints (..., depth, heigt, width)
        The shape of one patch. Can be a single int for all axis.

    num_patches : int or float, default=None
        The maximum number of patches to extract. If `max_patches` is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.

    Returns
    -------
    patches : array of shape (n_patches, patch_height, patch_width) or \
        (n_patches, patch_height, patch_width, n_channels)
        The collection of patches extracted from the image, where `n_patches`
        is either `max_patches` or the total number of patches that can be
        extracted.
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


def prod(atuple: Tuple[Union[float, int]]):
    # In python 3.8 there is a prod function in math, until then we have:
    return reduce(mul, atuple)
