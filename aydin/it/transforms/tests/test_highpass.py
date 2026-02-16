"""Tests for the highpass transform."""

import numpy
from skimage.data import binary_blobs
from skimage.util import random_noise

from aydin.io.datasets import camera, normalise
from aydin.it.transforms.highpass import HighpassTransform


def test_high_pass():
    """Test highpass transform preprocess/postprocess round-trip on a 2D image."""
    image = normalise(camera())
    image = random_noise(image, mode="s&p", amount=0.03, rng=0, clip=False)

    ac = HighpassTransform()

    preprocessed = ac.preprocess(image)
    postprocessed = ac.postprocess(preprocessed)

    assert image.shape == postprocessed.shape
    assert image.dtype == postprocessed.dtype

    assert postprocessed.dtype == image.dtype
    assert numpy.abs(postprocessed - image).mean() < 1e-8


def test_high_pass_3d():
    """Test highpass transform on 3D volume."""
    image = binary_blobs(length=32, n_dim=3, rng=1).astype(numpy.float32)
    image = random_noise(image, mode="s&p", amount=0.03, rng=0, clip=False).astype(
        numpy.float32
    )

    ac = HighpassTransform()

    preprocessed = ac.preprocess(image)
    postprocessed = ac.postprocess(preprocessed)

    assert preprocessed.shape == image.shape
    assert postprocessed.shape == image.shape
    assert postprocessed.dtype == image.dtype
    assert numpy.abs(postprocessed - image).mean() < 1e-8
