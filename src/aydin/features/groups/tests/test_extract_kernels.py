"""Tests for extract_patches_nd from aydin.util.patch_transform.patch_transform."""

import numpy

from aydin.util.patch_transform.patch_transform import extract_patches_nd


def test_extract_patches_nd_2d_all():
    """Test extracting all patches from a small 2D image."""
    image = numpy.arange(16, dtype=numpy.float32).reshape(4, 4)
    patch_size = 3

    patches = extract_patches_nd(image, patch_size=patch_size)

    # For a 4x4 image with patch_size=3, we expect (4-3+1)*(4-3+1) = 4 patches
    assert patches.shape == (4, 3, 3)

    # The first patch should be the top-left 3x3 block:
    expected_first = image[:3, :3]
    assert numpy.array_equal(patches[0], expected_first)


def test_extract_patches_nd_2d_max_patches():
    """Test extracting a limited number of patches from a 2D image."""
    rng = numpy.random.RandomState(0)
    image = rng.rand(32, 32).astype(numpy.float32)
    patch_size = 5
    max_patches = 10

    patches = extract_patches_nd(image, patch_size=patch_size, max_patches=max_patches)

    # Should return at most max_patches patches:
    assert patches.shape[0] <= max_patches
    # Each patch should have the correct shape:
    assert patches.shape[1:] == (5, 5)


def test_extract_patches_nd_3d():
    """Test extracting patches from a 3D volume."""
    rng = numpy.random.RandomState(42)
    volume = rng.rand(8, 8, 8).astype(numpy.float32)
    patch_size = 3

    patches = extract_patches_nd(volume, patch_size=patch_size)

    # For 8x8x8 with patch_size=3: (8-3+1)^3 = 216 patches
    expected_count = (8 - 3 + 1) ** 3
    assert patches.shape == (expected_count, 3, 3, 3)


def test_extract_patches_nd_tuple_patch_size():
    """Test extracting patches with a tuple patch size (non-square)."""
    image = numpy.arange(20, dtype=numpy.float32).reshape(4, 5)
    patch_size = (2, 3)

    patches = extract_patches_nd(image, patch_size=patch_size)

    # (4-2+1) * (5-3+1) = 3 * 3 = 9 patches
    assert patches.shape == (9, 2, 3)


def test_extract_patches_nd_patch_values_correct():
    """Test that extracted patch values are correct subsets of the image."""
    image = numpy.arange(25, dtype=numpy.float32).reshape(5, 5)
    patch_size = 3

    patches = extract_patches_nd(image, patch_size=patch_size)

    # Check a specific known patch (top-left corner):
    expected = numpy.array([[0, 1, 2], [5, 6, 7], [10, 11, 12]], dtype=numpy.float32)
    assert numpy.array_equal(patches[0], expected)

    # Check another known patch (row=1, col=1 -> second row, second col in grid):
    # Grid has (5-3+1)=3 columns, so patch at grid (1,1) is index 1*3+1=4
    expected_11 = numpy.array(
        [[6, 7, 8], [11, 12, 13], [16, 17, 18]], dtype=numpy.float32
    )
    assert numpy.array_equal(patches[4], expected_11)


def test_extract_patches_nd_deterministic_with_seed():
    """Test that random patch extraction is deterministic given a random_state."""
    rng = numpy.random.RandomState(0)
    image = rng.rand(32, 32).astype(numpy.float32)

    patches1 = extract_patches_nd(image, patch_size=5, max_patches=10, random_state=123)
    patches2 = extract_patches_nd(image, patch_size=5, max_patches=10, random_state=123)

    assert numpy.array_equal(patches1, patches2)
