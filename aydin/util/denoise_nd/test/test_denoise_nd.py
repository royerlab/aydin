# flake8: noqa
import numpy
from scipy.ndimage import gaussian_filter

from aydin.util.denoise_nd.denoise_nd import extend_nd


def test_denoise_nd():
    # raw function that only supports 2D images:
    def function(image, sigma):
        if image.ndim != 2:
            raise RuntimeError("Function only supports arrays of dimensions 2")
        return gaussian_filter(image, sigma)

    # extended function that supports all dimension (with all caveats associated to how we actually do this extension...)
    @extend_nd(available_dims=[2])
    def extended_function(image, sigma):
        return function(image, sigma)

    # Wrongly extended function: we pretend that it can do dim 1 when in fact it can't!
    @extend_nd(available_dims=[1, 2])
    def wrongly_extended_function(image, sigma):
        return function(image, sigma)

    image = numpy.zeros((32,))
    image[16] = 1

    try:
        function(image, sigma=1)
        assert False
    except RuntimeError as e:
        # expected!
        assert True

    try:
        extended_function(image, sigma=1)
        assert True
    except RuntimeError as e:
        assert False

    try:
        wrongly_extended_function(image, sigma=1)
        assert False
    except RuntimeError as e:
        assert True

    image = numpy.zeros((32, 5, 64))
    image[16, 2, 32] = 1

    try:
        function(image, sigma=1)
        assert False
    except RuntimeError as e:
        # expected!
        assert True

    try:
        extended_function(image, sigma=1)
        assert True
    except RuntimeError as e:
        assert False
