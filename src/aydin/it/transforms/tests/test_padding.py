"""Tests for the padding transform."""

import numpy
from skimage.data import binary_blobs

from aydin.io.datasets import normalise
from aydin.it.transforms.padding import PaddingTransform


def test_padding():
    """Test padding transform preprocess/postprocess round-trip on a 3D image."""
    image = binary_blobs(length=128, rng=1, n_dim=3).astype(numpy.float32)
    image = normalise(image)

    pt = PaddingTransform(pad_width=17)

    preprocessed = pt.preprocess(image)
    postprocessed = pt.postprocess(preprocessed)

    assert postprocessed.dtype == image.dtype
    assert postprocessed.shape == image.shape
    assert numpy.abs(postprocessed - image).mean() < 1e-8


def test_padding_odd_dimensions():
    """Test padding on an image with odd dimensions."""
    image = numpy.random.RandomState(42).random((33, 47)).astype(numpy.float32)

    pt = PaddingTransform(pad_width=8)

    preprocessed = pt.preprocess(image)
    postprocessed = pt.postprocess(preprocessed)

    assert postprocessed.shape == image.shape
    assert postprocessed.dtype == image.dtype
    assert numpy.abs(postprocessed - image).mean() < 1e-8


def test_padding_2d():
    """Test padding transform on a 2D image."""
    image = numpy.random.RandomState(42).random((64, 64)).astype(numpy.float32)

    pt = PaddingTransform(pad_width=16)

    preprocessed = pt.preprocess(image)
    # Preprocessed should be larger
    assert preprocessed.shape[0] >= image.shape[0]
    assert preprocessed.shape[1] >= image.shape[1]

    postprocessed = pt.postprocess(preprocessed)
    assert postprocessed.shape == image.shape
    assert numpy.abs(postprocessed - image).mean() < 1e-8
