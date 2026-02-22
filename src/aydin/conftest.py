"""Shared pytest fixtures for the aydin test suite."""

import numpy
import pytest
from skimage.data import binary_blobs, camera


@pytest.fixture(scope='session')
def sample_2d_uint8_image():
    """Session-scoped uint8 camera image for dtype-specific tests."""
    return camera()


@pytest.fixture(scope='session')
def sample_3d_image():
    """Session-scoped 3D float32 test volume (32x64x64 binary blobs)."""
    return binary_blobs(length=64, n_dim=3, rng=1).astype(numpy.float32)


@pytest.fixture
def tmp_model_path(tmp_path):
    """Function-scoped temp directory for save/load tests."""
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    return str(model_dir)
