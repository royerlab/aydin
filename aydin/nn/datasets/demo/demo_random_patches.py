"""Demo script illustrating random patch extraction from images.

Demonstrates the use of ``random_sample_patches`` for extracting
informative patches from a 3D HCR image volume.
"""

import numpy

from aydin.io.datasets import camera
from aydin.nn.datasets.random_patches import random_patches


def demo_nb_patches_per_image(nb_patches_per_image):
    """Extract random patches from a camera image and report counts.

    Parameters
    ----------
    nb_patches_per_image : int
        Number of patches to extract per image.
    """
    image_array = camera()

    image_array = numpy.expand_dims(image_array, 0)
    image_array = numpy.expand_dims(image_array, 0)

    patch_slicing_objects = random_patches(
        image=image_array,
        patch_size=16,
        nb_patches_per_image=nb_patches_per_image,
    )

    print(len(patch_slicing_objects), nb_patches_per_image)
    print(patch_slicing_objects[0])


if __name__ == '__main__':
    for nb in [1, 10, 100, 1000]:
        demo_nb_patches_per_image(nb)
