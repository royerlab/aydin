import numpy

from aydin.io.datasets import camera
from aydin.nn.datasets.random_patches import random_patches


def demo_nb_patches_per_image(nb_patches_per_image):
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
