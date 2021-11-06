import numpy
from scipy.ndimage import shift
from skimage.exposure import rescale_intensity

from aydin.util.fast_shift.fast_shift import fast_shift
from aydin.io.datasets import newyork, examples_single


def _normalise(image):
    return rescale_intensity(
        image.astype(numpy.float32, copy=False), in_range='image', out_range=(0, 1)
    )


def test_fast_shift_filter_type_support():
    image = newyork()

    _run_test_for_type(image.astype(dtype=numpy.float32))
    _run_test_for_type(image.astype(dtype=numpy.float16), decimal=1)
    _run_test_for_type(image.astype(dtype=numpy.uint32), decimal=0)
    _run_test_for_type(image.astype(dtype=numpy.uint16), decimal=0)
    _run_test_for_type(image.astype(dtype=numpy.uint8), decimal=0)


def _run_test_for_type(image, decimal=3, _shift=(-1, 3)):
    shifted_image = fast_shift(image, shift=_shift)
    assert shifted_image.dtype == image.dtype

    shifted_image = shifted_image.astype(dtype=numpy.float32, copy=False)
    scipy_shifted_image = shift(
        image.astype(dtype=numpy.float32, copy=False), shift=_shift, mode="constant"
    ).astype(dtype=numpy.float32, copy=False)
    numpy.testing.assert_array_almost_equal(
        shifted_image, scipy_shifted_image, decimal=decimal
    )


def test_compute_uniform_filter_different_sizes():

    shifts = [(1, 2), (16, 7), (3, 7), (4, 4)]

    for current_shift in shifts:
        test_fast_shift_filter_2d(_shift=current_shift)


def test_fast_shift_filter_1d(_shift=(-1,)):
    image = _normalise(newyork().astype(numpy.float32))
    image = image[512, :]

    shifted_image = fast_shift(image, shift=_shift)
    scipy_shifted_image = shift(image, shift=_shift, mode="constant")

    numpy.testing.assert_array_almost_equal(
        shifted_image, scipy_shifted_image, decimal=3
    )


def test_fast_shift_filter_2d(_shift=(-1, 3)):
    image = _normalise(newyork().astype(numpy.float32))

    shifted_image = fast_shift(image, shift=_shift)
    scipy_shifted_image = shift(image, shift=_shift, mode="constant")

    numpy.testing.assert_array_almost_equal(
        shifted_image, scipy_shifted_image, decimal=3
    )


def test_fast_shift_filter_3d(_shift=(-1, 3, -7)):
    islet = examples_single.royerlab_hcr.get_array().squeeze()
    image = islet[2, :60, 0:256, 0:256]

    shifted_image = fast_shift(image, shift=_shift)
    scipy_shifted_image = shift(image, shift=_shift, mode="constant")

    numpy.testing.assert_array_almost_equal(
        shifted_image, scipy_shifted_image, decimal=3
    )


def test_fast_shift_filter_4d(_shift=(-1, 3, -7, +13)):
    image = examples_single.hyman_hela.get_array().squeeze()
    image = image[0:10, 0:10, 0:128, 0:128]

    shifted_image = fast_shift(image, shift=_shift)
    scipy_shifted_image = shift(image, shift=_shift, mode="constant")

    numpy.testing.assert_array_almost_equal(
        shifted_image, scipy_shifted_image, decimal=3
    )
