import numpy
from numba import jit
from skimage.metrics import structural_similarity


@jit(nopython=True, parallel=True)
def mean_squared_error(image0, image1):
    return numpy.mean((image0 - image1) ** 2)


@jit(nopython=True, parallel=True)
def mean_absolute_error(image_a, image_b):
    return numpy.mean(numpy.absolute(image_a - image_b))


@jit(nopython=True, parallel=True)
def lhalf_error(image_a, image_b):
    return numpy.mean(numpy.absolute(image_a - image_b) ** 0.5) ** 2


def structural_loss(image_a, image_b):
    return numpy.astype(
        1 - structural_similarity(image_a, image_b), dtype=numpy.float32
    )
