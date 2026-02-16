"""Tests for aydin.io.datasets module."""

import os
from unittest.mock import patch

import numpy

from aydin.io.datasets import (
    add_blur_2d,
    add_blur_3d,
    add_noise,
    camera,
    cropped_newyork,
    dots,
    download_from_gdrive,
    examples_single,
    lizard,
    newyork,
    normalise,
    pollen,
    small_newyork,
)


def test_normalise():
    """Test that normalise produces values in [0, 1] range."""
    raw_image = numpy.array([[0, 100, 200], [50, 150, 255]], dtype=numpy.uint8)
    normalized = normalise(raw_image)

    assert normalized.dtype == numpy.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert numpy.isclose(normalized.min(), 0.0, atol=1e-6)
    assert numpy.isclose(normalized.max(), 1.0, atol=1e-6)


def test_normalise_already_normalized():
    """Test normalise on already normalized data."""
    image = numpy.array([[0.0, 0.5], [0.25, 1.0]], dtype=numpy.float32)
    normalized = normalise(image)

    assert normalized.dtype == numpy.float32
    assert numpy.allclose(normalized, image)


def test_add_noise_reproducibility():
    """Test that seed produces reproducible results."""
    image = normalise(numpy.random.rand(64, 64).astype(numpy.float32))

    noisy1 = add_noise(image, seed=42)
    noisy2 = add_noise(image, seed=42)

    assert numpy.array_equal(noisy1, noisy2)


def test_add_noise_different_seeds():
    """Test that different seeds produce different results."""
    image = normalise(numpy.random.rand(64, 64).astype(numpy.float32))

    noisy1 = add_noise(image, seed=42)
    noisy2 = add_noise(image, seed=123)

    assert not numpy.array_equal(noisy1, noisy2)


def test_add_noise_parameters():
    """Test add_noise with various parameter combinations."""
    image = normalise(numpy.random.rand(32, 32).astype(numpy.float32))

    # Test with only Gaussian noise (no Poisson)
    noisy = add_noise(image, intensity=None, variance=0.01, sap=0.0, seed=42)
    assert noisy.shape == image.shape
    assert noisy.dtype == numpy.float32

    # Test with higher variance
    noisy_high = add_noise(image, intensity=None, variance=0.1, sap=0.0, seed=42)
    # Higher variance should produce more deviation from original
    diff_high = numpy.abs(noisy_high - image).mean()
    diff_low = numpy.abs(noisy - image).mean()
    assert diff_high > diff_low


def test_camera_dataset():
    """Test camera image loader."""
    image = camera()

    assert image is not None
    assert isinstance(image, numpy.ndarray)
    assert image.ndim == 2
    assert image.shape[0] > 0 and image.shape[1] > 0


def test_lizard_dataset():
    """Test lizard image loader."""
    image = lizard()

    assert image is not None
    assert isinstance(image, numpy.ndarray)
    assert image.ndim == 2


def test_newyork_dataset():
    """Test newyork image loader."""
    image = newyork()

    assert image is not None
    assert isinstance(image, numpy.ndarray)
    assert image.ndim == 2
    assert image.shape == (1024, 1024)


def test_small_newyork():
    """Test small_newyork returns downscaled image."""
    image = small_newyork()

    assert image is not None
    assert isinstance(image, numpy.ndarray)
    assert image.ndim == 2
    # Should be half the size of newyork (1024x1024 -> 512x512)
    assert image.shape == (512, 512)


def test_cropped_newyork():
    """Test cropped_newyork returns center-cropped image."""
    image = cropped_newyork(crop_amount=256)

    assert image is not None
    assert isinstance(image, numpy.ndarray)
    # Original is 1024x1024, cropped by 256 on each side
    assert image.shape == (512, 512)


def test_cropped_newyork_limits_crop():
    """Test that crop_amount is limited to max 500."""
    image = cropped_newyork(crop_amount=1000)

    assert image is not None
    # Should be limited to 500, so result is 24x24
    assert image.shape == (24, 24)


def test_pollen_dataset():
    """Test pollen image loader."""
    image = pollen()

    assert image is not None
    assert isinstance(image, numpy.ndarray)
    assert image.ndim == 2


def test_add_blur_2d():
    """Test 2D blur function."""
    image = numpy.random.rand(64, 64).astype(numpy.float32)

    blurred, psf_kernel = add_blur_2d(image, dz=0)

    assert blurred.shape == image.shape
    assert psf_kernel is not None
    assert psf_kernel.ndim == 2
    # PSF should be normalized (sum to 1)
    assert numpy.isclose(psf_kernel.sum(), 1.0, atol=1e-6)


def test_add_blur_2d_different_dz():
    """Test 2D blur with different dz offsets."""
    image = numpy.random.rand(64, 64).astype(numpy.float32)

    blurred1, _ = add_blur_2d(image, dz=0)
    blurred2, _ = add_blur_2d(image, dz=4)

    # Different dz should produce different blurs
    assert not numpy.array_equal(blurred1, blurred2)


def test_add_blur_3d():
    """Test 3D blur function."""
    image = numpy.random.rand(16, 32, 32).astype(numpy.float32)

    blurred, psf_kernel = add_blur_3d(image, xy_size=9, z_size=9)

    assert blurred.shape == image.shape
    assert psf_kernel is not None
    assert psf_kernel.ndim == 3
    # PSF should be normalized (sum to 1)
    assert numpy.isclose(psf_kernel.sum(), 1.0, atol=1e-6)


def test_dots_synthetic_image():
    """Test dots() synthetic image generator."""
    image = dots()

    assert image is not None
    assert isinstance(image, numpy.ndarray)
    assert image.shape == (512, 512)
    assert image.dtype == numpy.float32
    # Image should have values in reasonable range
    assert image.min() >= 0
    assert image.max() <= 1


def test_dots_has_sparse_structure():
    """Test that dots image is sparse (mostly background)."""
    image = dots()

    # Most pixels should be near 0 or 0.1 (background)
    low_value_fraction = (image < 0.5).mean()
    assert low_value_fraction > 0.9  # At least 90% should be low values


def test_examples_single_enum():
    """Test that examples_single enum has expected members."""
    # Check a few known members exist
    assert hasattr(examples_single, 'generic_camera')
    assert hasattr(examples_single, 'generic_lizard')
    assert hasattr(examples_single, 'generic_newyork')
    assert hasattr(examples_single, 'noisy_fountain')


def test_examples_single_get_path():
    """Test examples_single.get_path() returns valid path."""
    path = examples_single.generic_lizard.get_path()

    assert path is not None
    assert isinstance(path, str)
    assert os.path.exists(path)
    assert path.endswith('.png')


def test_examples_single_get_array():
    """Test examples_single.get_array() returns numpy array."""
    array = examples_single.generic_lizard.get_array()

    assert array is not None
    assert isinstance(array, numpy.ndarray)


@patch('aydin.io.datasets.gdown.download')
@patch('aydin.io.datasets.exists')
def test_download_from_gdrive_already_exists(mock_exists, mock_download):
    """Test download_from_gdrive skips when file exists."""
    mock_exists.return_value = True

    result = download_from_gdrive('fake_id', 'test.tif', '/tmp', overwrite=False)

    assert result is None
    mock_download.assert_not_called()


@patch('aydin.io.datasets.gdown.download')
@patch('aydin.io.datasets.exists')
def test_download_from_gdrive_downloads_when_missing(mock_exists, mock_download):
    """Test download_from_gdrive downloads when file doesn't exist."""
    mock_exists.return_value = False

    result = download_from_gdrive('fake_id', 'test.tif', '/tmp', overwrite=False)

    assert result == '/tmp/test.tif'
    mock_download.assert_called_once()


@patch('aydin.io.datasets.gdown.download')
@patch('aydin.io.datasets.exists')
def test_download_from_gdrive_overwrite(mock_exists, mock_download):
    """Test download_from_gdrive re-downloads when overwrite=True."""
    mock_exists.return_value = True

    result = download_from_gdrive('fake_id', 'test.tif', '/tmp', overwrite=True)

    assert result == '/tmp/test.tif'
    mock_download.assert_called_once()
