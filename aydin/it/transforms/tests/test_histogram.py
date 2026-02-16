"""Tests for the histogram equalisation transform."""

import numpy

from aydin.io.datasets import camera
from aydin.it.transforms.histogram import HistogramEqualisationTransform


def test_histogram():
    """Test histogram equalisation preprocess/postprocess round-trip on a 2D image."""
    image = camera()

    ht = HistogramEqualisationTransform()

    preprocessed = ht.preprocess(image)
    postprocessed = ht.postprocess(preprocessed)

    assert postprocessed.dtype == image.dtype
    assert postprocessed.shape == image.shape
    assert (
        numpy.abs(
            postprocessed.astype(numpy.float32) - image.astype(numpy.float32)
        ).mean()
        < 2
    )


def test_histogram_3d():
    """Test histogram equalization transform on 3D volume."""
    # Use gradient data instead of binary blobs for more realistic histogram
    x = numpy.linspace(0, 255, 32)
    image = (
        x[:, numpy.newaxis, numpy.newaxis]
        + x[numpy.newaxis, :, numpy.newaxis] * 0.5
        + x[numpy.newaxis, numpy.newaxis, :] * 0.25
    )
    image = numpy.clip(image, 0, 255).astype(numpy.uint8)

    ht = HistogramEqualisationTransform()

    preprocessed = ht.preprocess(image)
    postprocessed = ht.postprocess(preprocessed)

    assert preprocessed.shape == image.shape
    assert postprocessed.dtype == image.dtype
    assert postprocessed.shape == image.shape
    assert (
        numpy.abs(
            postprocessed.astype(numpy.float32) - image.astype(numpy.float32)
        ).mean()
        < 2
    )
