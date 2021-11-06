from typing import Tuple, Union
import numpy
from pynndescent import NNDescent

from aydin.features.groups.extract_kernels import extract_patches_nd
from aydin.util.patch_transform.patch_transform import reconstruct_from_nd_patches


def nearest_self_image(image, patch_shape: Union[int, Tuple[int, ...]] = 5):
    """
    This function computes the 'nearest self image' which is obtained by
    replacing for each patch the nearest patch within the same image.

    Parameters
    ----------
    image: ArrayLike
        Nearest image to build.
    """

    # Normalise patch shape:
    if type(patch_shape) is not tuple:
        patch_shape = (patch_shape,) * image.ndim

    # Let's ensure the image is a 32 bit float image
    image = image.astype(dtype=numpy.float32)

    # First we need to pad the image.
    # By how much? this depends on how much low filtering we need to do:
    pad_width = tuple((ps // 2, ps // 2) for ps in patch_shape)

    # pad image:
    image = numpy.pad(image, pad_width=pad_width, mode='reflect')

    # Let's save the shape of the padded image:
    padded_image_shape = image.shape

    # Extract patches for the image:
    patches = extract_patches_nd(image, patch_shape=patch_shape)

    # reshape patches as vectors:
    original_patches_shape = patches.shape
    patches = patches.reshape(patches.shape[0], -1)

    # Build the data structure for nearest neighbor search:
    nn = NNDescent(patches, n_jobs=-1)

    # for each patch we query for the nearest patch:
    k = 4
    indices, distances = nn.query(patches, k=k)

    for i in range(len(patches)):
        for j in range(k):
            if indices[i, j] != i:
                indices[i, 0] = indices[i, j]
                break

    patches = patches[indices[:, 0]]

    # scikit-learn version:
    # # Build the kd-tree for nearest neighbor search:
    # tree = KDTree(patches, leaf_size=4)
    #
    # # for each patch we query for the nearest patch:
    # distances, indices = tree.query(patches, k=2)
    #
    # patches = patches[indices[:, 1]]

    # reshape patches back to their original shape:
    patches = patches.reshape(original_patches_shape)

    # Transform back from patches to image:
    reconstructed_image = reconstruct_from_nd_patches(
        patches, padded_image_shape, mode='center'
    )

    # Crop to remove padding:
    reconstructed_image = reconstructed_image[tuple(slice(u, -v) for u, v in pad_width)]

    return reconstructed_image
