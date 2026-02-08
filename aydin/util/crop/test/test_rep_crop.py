# flake8: noqa

import numpy
import pytest

from aydin.io.datasets import newyork
from aydin.util.crop.demo.demo_rep_crop import demo_representative_crop
from aydin.util.crop.demo.demo_sf_rep_crop import demo_super_fast_representative_crop
from aydin.util.crop.rep_crop import representative_crop


def test_representative_crop():
    newyork_image = newyork()
    demo_representative_crop(newyork_image, display=False)


def test_super_fast_representative_crop():
    newyork_image = newyork()
    demo_super_fast_representative_crop(newyork_image, display=False)


def test_representative_crop_small_batch_dimension():
    """Regression test for GitHub issue #207.

    A ZeroDivisionError occurred when the image had a small batch dimension
    (e.g., shape (2, 84, 580, 576)) because the granularity for that
    dimension was computed as 0 (cs // granularity_factor where cs < factor).
    """
    numpy.random.seed(42)
    # The exact shape from the bug report
    image = numpy.random.rand(2, 84, 580, 576).astype(numpy.float32)
    crop = representative_crop(image, crop_size=int(1e6), favour_odd_lengths=True)
    assert crop is not None
    assert crop.ndim == 4


@pytest.mark.parametrize(
    "shape",
    [
        (1, 64, 64),
        (3, 3, 64, 64),
        (2, 2, 2, 32, 32),
        (1, 1, 128, 128),
    ],
)
def test_representative_crop_small_leading_dimensions(shape):
    """Ensure representative_crop handles images with small leading
    dimensions (common for batch/channel axes) without errors."""
    numpy.random.seed(42)
    image = numpy.random.rand(*shape).astype(numpy.float32)
    crop = representative_crop(image, crop_size=int(1e5))
    assert crop is not None
    assert crop.ndim == len(shape)


def test_representative_crop_dimension_equals_crop():
    """Test that no error occurs when a dimension exactly equals the
    cropped size (the cs == s edge case that caused the original bug)."""
    numpy.random.seed(42)
    # Small image where crop_size is larger than the image itself
    image = numpy.random.rand(32, 32).astype(numpy.float32)
    crop = representative_crop(image, crop_size=32 * 32 + 1)
    assert crop is not None
    # When image is already small enough, the whole image is returned
    assert crop.shape == image.shape
