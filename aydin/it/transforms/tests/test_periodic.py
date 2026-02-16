"""Tests for the periodic noise suppression transform."""

import numpy
from skimage.data import binary_blobs

from aydin.io.datasets import add_noise, camera, normalise
from aydin.it.transforms.periodic import PeriodicNoiseSuppressionTransform


def test_high_pass():
    """Test periodic noise suppression preprocess/postprocess round-trip."""
    image = normalise(camera().astype(numpy.float32))

    freq = 96
    periodic_pattern = 0.3 * (
        1 + numpy.cos(numpy.linspace(0, freq * 2 * numpy.pi, num=image.shape[0]))
    )
    periodic_pattern = periodic_pattern[:, numpy.newaxis]
    image += periodic_pattern

    image = add_noise(image)

    pns = PeriodicNoiseSuppressionTransform(post_processing_is_inverse=True)

    preprocessed = pns.preprocess(image)
    postprocessed = pns.postprocess(preprocessed)

    assert image.shape == postprocessed.shape
    assert image.dtype == postprocessed.dtype

    assert postprocessed.dtype == image.dtype
    assert numpy.abs(postprocessed - image).mean() < 1e-2


def test_periodic_3d():
    """Test periodic noise suppression transform on 3D volume."""
    image = binary_blobs(length=32, n_dim=3, rng=1).astype(numpy.float32)
    image = normalise(image)

    # Add periodic pattern along one axis
    freq = 8
    periodic_pattern = 0.3 * (
        1 + numpy.cos(numpy.linspace(0, freq * 2 * numpy.pi, num=image.shape[0]))
    )
    periodic_pattern = periodic_pattern[:, numpy.newaxis, numpy.newaxis]
    image += periodic_pattern

    image = add_noise(image)

    pns = PeriodicNoiseSuppressionTransform(post_processing_is_inverse=True)

    preprocessed = pns.preprocess(image)
    postprocessed = pns.postprocess(preprocessed)

    assert preprocessed.shape == image.shape
    assert postprocessed.shape == image.shape
    assert postprocessed.dtype == image.dtype
    assert numpy.abs(postprocessed - image).mean() < 1e-2
