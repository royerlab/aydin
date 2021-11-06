import numpy
from skimage.exposure import rescale_intensity

from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.cnn.util.data_util import random_sample_patches
import napari


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def demo():
    # Load image
    image_path = examples_single.royerlab_hcr.get_path()
    image0, metadata = io.imread(image_path)
    print(image0.shape)
    image0 = n(image0.squeeze())

    image0 = numpy.expand_dims(image0[1:2], -1)
    tile_size = (64, 64, 64)
    num_tile = 500
    adoption_rate = 0.5
    input_data = random_sample_patches(image0, tile_size, num_tile, adoption_rate)

    imgpatch_int = numpy.zeros(image0.shape)
    for i in input_data:
        imgpatch_int[i[:-1]] += 1
    image0 = rescale_intensity(image0.squeeze().astype(numpy.float64), out_range=(0, 1))
    imgpatch_int = rescale_intensity(imgpatch_int.squeeze(), out_range=(0, 1))
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image0)
        viewer.add_image(imgpatch_int)


demo()
