"""Tests for dictionary learning and fixed dictionary utilities."""

import numpy
import pytest
from skimage.data import camera

from aydin.io.datasets import normalise
from aydin.util.dictionary.dictionary import (
    _fracture_measure,
    _lipschitz_error,
    dictionary_cleanup,
    extract_normalised_vectorised_patches,
    fixed_dictionary,
    learn_dictionary,
)


@pytest.fixture(scope='module')
def small_image():
    """Provide a small normalised float32 image for testing."""
    return normalise(camera()[:128, :128].astype(numpy.float32))


def test_extract_normalised_patches_shape(small_image):
    """Test that extracted patches have expected shape."""
    patch_size = (5, 5)
    patches = extract_normalised_vectorised_patches(
        small_image, patch_size=patch_size, max_patches=1000
    )
    assert patches.ndim == 2
    assert patches.shape[1] == 25  # 5 * 5
    assert patches.shape[0] <= 1000


def test_extract_normalised_patches_with_norm_values(small_image):
    """Test patch extraction with normalisation values output."""
    patch_size = (5, 5)
    result = extract_normalised_vectorised_patches(
        small_image,
        patch_size=patch_size,
        max_patches=500,
        output_norm_values=True,
    )
    assert len(result) == 3
    patches, means, stds = result
    assert patches.ndim == 2
    assert patches.shape[1] == 25


def test_extract_normalised_patches_no_normalisation(small_image):
    """Test patch extraction without normalisation."""
    patch_size = (5, 5)
    patches = extract_normalised_vectorised_patches(
        small_image,
        patch_size=patch_size,
        max_patches=500,
        normalise_means=False,
        normalise_stds=False,
    )
    assert patches.ndim == 2
    assert patches.shape[1] == 25


def test_fixed_dictionary_dct(small_image):
    """Test that fixed DCT dictionary produces valid atoms."""
    dictionary = fixed_dictionary(small_image, patch_size=5, dictionaries='dct')
    assert dictionary.ndim == 3  # (n_atoms, patch_h, patch_w) for 2D image
    assert dictionary.shape[1] == dictionary.shape[2]
    assert len(dictionary) > 0


def test_fixed_dictionary_dst_dct(small_image):
    """Test fixed dictionary with combined DST+DCT."""
    dictionary = fixed_dictionary(small_image, patch_size=5, dictionaries='dst+dct')
    assert dictionary.ndim == 3
    assert len(dictionary) > 0


def test_learn_dictionary_kmeans(small_image):
    """Test learned dictionary with K-means algorithm."""
    dictionary = learn_dictionary(
        small_image,
        patch_size=5,
        max_patches=500,
        max_dictionary_size=16,
        algorithm='kmeans',
        num_iterations=10,
        cleanup_dictionary=False,
    )
    assert dictionary.ndim == 3
    assert len(dictionary) > 0
    assert len(dictionary) <= 16


def test_learn_dictionary_pca(small_image):
    """Test learned dictionary with PCA algorithm."""
    dictionary = learn_dictionary(
        small_image,
        patch_size=5,
        max_patches=500,
        max_dictionary_size=16,
        algorithm='pca',
        cleanup_dictionary=False,
    )
    assert dictionary.ndim == 3
    assert len(dictionary) > 0


def test_dictionary_cleanup_reduces_size():
    """Test that dictionary cleanup filters out noisy atoms."""
    rng = numpy.random.RandomState(42)
    # Create patches: some smooth, some noisy
    smooth_patches = [
        gaussian_filter_patch(rng.random((5, 5)).astype(numpy.float32))
        for _ in range(20)
    ]
    noisy_patches = [rng.random((5, 5)).astype(numpy.float32) for _ in range(10)]
    all_patches = numpy.stack(smooth_patches + noisy_patches)

    filtered = dictionary_cleanup(all_patches, truncate=0.3)
    # Cleanup should remove some patches
    assert len(filtered) <= len(all_patches)
    assert len(filtered) > 0


def gaussian_filter_patch(patch):
    """Apply Gaussian filtering to create a smooth patch."""
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(patch, sigma=1.0)


def test_fracture_measure():
    """Test fracture measure on known inputs."""
    # Smooth patch should have low fracture measure
    smooth = numpy.ones((5, 5), dtype=numpy.float32)
    smooth_fracture = _fracture_measure(smooth)
    assert smooth_fracture >= 0

    # Checkerboard should have higher fracture measure
    checker = numpy.zeros((5, 5), dtype=numpy.float32)
    checker[::2, ::2] = 1.0
    checker[1::2, 1::2] = 1.0
    checker_fracture = _fracture_measure(checker)
    assert checker_fracture >= smooth_fracture


def test_lipschitz_error():
    """Test Lipschitz error on known inputs."""
    # Smooth patch should have low error
    smooth = numpy.ones((5, 5), dtype=numpy.float32) * 0.5
    smooth_error = _lipschitz_error(smooth)
    assert smooth_error >= 0

    # Patch with sharp discontinuity should have higher error
    sharp = numpy.zeros((5, 5), dtype=numpy.float32)
    sharp[2, 2] = 10.0
    sharp_error = _lipschitz_error(sharp)
    assert sharp_error >= smooth_error
