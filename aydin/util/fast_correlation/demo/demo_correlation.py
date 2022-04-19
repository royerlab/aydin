# flake8: noqa
import numpy
from numpy import ones, absolute
from numpy.random import rand
from scipy.ndimage import correlate as scipy_ndimage_correlate

from aydin.io.datasets import examples_single, newyork
from aydin.util.fast_correlation.correlation import correlate
from aydin.util.log.log import lsection, Log


def demo_convolution(image):
    Log.enable_output = True

    kernel_shape = (3, 5, 7, 9, 11, 3)[: image.ndim]
    kernel = ones(kernel_shape)
    kernel /= kernel.sum()

    # warmup:
    correlate(image, kernel)

    with lsection(f"correlation..."):
        convolved = correlate(image, kernel)

    with lsection(f"Reference correlation:... "):
        convolved_ref = scipy_ndimage_correlate(image, kernel)

    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(image, name='image', colormap='plasma')
    # viewer.add_image(convolved_ref, name='convolved_ref', colormap='plasma')
    # viewer.add_image(convolved, name='convolved', colormap='plasma')
    # viewer.add_image(absolute(convolved-convolved_ref), name='abs difference', colormap='plasma')
    # napari.run()


if __name__ == "__main__":
    image_2d = newyork().astype(numpy.float32)
    image_2d = image_2d[0:731, 0:897]
    demo_convolution(image_2d)

    image_3d = examples_single.royerlab_hcr.get_array().squeeze()
    image_3d = image_3d[:60, 2, 0 : 0 + 1524, 0 : 0 + 1524]
    image_3d = image_3d.astype(numpy.float32)
    demo_convolution(image_3d)

    image_4d = examples_single.hyman_hela.get_array().squeeze()
    image_4d = image_4d[..., 0:64, 0:64]
    image_4d = image_4d.astype(numpy.float32)
    demo_convolution(image_4d)

    image5d = rand(7, 6, 5, 7, 2)
    demo_convolution(image5d)

    image6d = rand(7, 8, 5, 6, 3, 5)
    demo_convolution(image6d)
