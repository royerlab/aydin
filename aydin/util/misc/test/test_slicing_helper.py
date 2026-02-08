import numpy
from skimage.data import camera

from aydin.util.misc.slicing_helper import apply_slicing


def test_apply_slicing():
    image = camera().astype(numpy.float32)

    assert numpy.array_equal(
        image[10:20, 32:57], apply_slicing(image, "[10:20, 32:57]")
    )
