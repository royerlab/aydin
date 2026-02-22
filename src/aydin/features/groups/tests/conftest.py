"""Shared fixtures for feature group tests."""

import numpy
import pytest
from skimage.data import camera


@pytest.fixture
def normalized_camera_image():
    """Return the skimage camera image normalized to float32 in [0, 1]."""
    return camera().astype(numpy.float32) / numpy.iinfo(numpy.uint8).max
