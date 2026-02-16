"""Shared test fixtures for CLI tests."""

import numpy
import pytest

from aydin.io.io import imwrite


@pytest.fixture
def paired_image_files(tmp_path):
    """Write a clean + noisy TIFF pair and return their paths."""
    rng = numpy.random.RandomState(42)
    clean = rng.uniform(0.2, 0.8, (64, 64)).astype(numpy.float32)
    noisy = (clean + rng.normal(0, 0.05, clean.shape)).astype(numpy.float32)

    clean_path = str(tmp_path / "clean.tif")
    noisy_path = str(tmp_path / "noisy.tif")
    imwrite(clean, clean_path)
    imwrite(noisy, noisy_path)
    return clean_path, noisy_path


@pytest.fixture
def single_image_file(tmp_path):
    """Write a single TIFF and return its path."""
    rng = numpy.random.RandomState(42)
    image = rng.uniform(0.2, 0.8, (64, 64)).astype(numpy.float32)

    path = str(tmp_path / "image.tif")
    imwrite(image, path)
    return path
