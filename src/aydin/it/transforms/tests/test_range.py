"""Tests for the range normalisation transform."""

import numpy
from skimage.data import binary_blobs

from aydin.io.datasets import newyork
from aydin.it.transforms.range import RangeTransform


def test_range_minmax():
    """Test range transform with min-max mode."""
    do_test_range("minmax")


def test_range_percentile():
    """Test range transform with percentile mode."""
    do_test_range("percentile")


def do_test_range(mode):
    """Run range transform preprocess/postprocess round-trip test.

    Parameters
    ----------
    mode : str
        Range normalisation mode ('minmax' or 'percentile').
    """
    image = newyork()

    rt = RangeTransform(mode=mode)

    preprocessed = rt.preprocess(image)
    postprocessed = rt.postprocess(preprocessed)

    assert postprocessed.dtype == image.dtype
    assert postprocessed.shape == image.shape
    assert numpy.abs(postprocessed - image).mean() < 1e-8


def test_range_3d():
    """Test range transform on 3D volume."""
    image = binary_blobs(length=32, n_dim=3, rng=1).astype(numpy.float32)

    for mode in ["minmax", "percentile"]:
        rt = RangeTransform(mode=mode)

        preprocessed = rt.preprocess(image)
        postprocessed = rt.postprocess(preprocessed)

        assert preprocessed.shape == image.shape
        assert postprocessed.dtype == image.dtype
        assert postprocessed.shape == image.shape
        assert numpy.abs(postprocessed - image).mean() < 1e-8


def test_range_constant_image():
    """Test range transform on an all-constant image (edge case)."""
    image = numpy.ones((32, 32), dtype=numpy.float32) * 42.0

    for mode in ["minmax", "percentile"]:
        rt = RangeTransform(mode=mode)
        preprocessed = rt.preprocess(image)
        postprocessed = rt.postprocess(preprocessed)

        assert postprocessed.shape == image.shape
        assert postprocessed.dtype == image.dtype
        # All values should be the same after round-trip
        assert numpy.allclose(postprocessed, image)


def test_range_single_pixel():
    """Test range transform on a single-pixel image."""
    image = numpy.array([[0.5]], dtype=numpy.float32)

    rt = RangeTransform(mode="minmax")
    preprocessed = rt.preprocess(image)
    postprocessed = rt.postprocess(preprocessed)

    assert postprocessed.shape == (1, 1)
    assert numpy.allclose(postprocessed, image)
