"""Tests for RandomMaskedDataset and GridMaskedDataset."""

import numpy
import torch

from aydin.nn.datasets.grid_masked_dataset import GridMaskedDataset
from aydin.nn.datasets.random_masked_dataset import RandomMaskedDataset


def test_random_masked_dataset_2d():
    """RandomMaskedDataset works with 2D images."""
    image = numpy.random.rand(1, 1, 64, 64).astype(numpy.float32)
    dataset = RandomMaskedDataset(image, patch_size=32)

    assert len(dataset) > 0

    original, input_patch, mask = dataset[0]
    assert original.shape == input_patch.shape
    assert original.shape[0] == 1  # channel dim
    assert mask.shape == original.shape


def test_random_masked_dataset_3d():
    """RandomMaskedDataset works with 3D images (regression test for conv2d bug)."""
    image = numpy.random.rand(1, 1, 16, 16, 16).astype(numpy.float32)
    dataset = RandomMaskedDataset(image, patch_size=16)

    assert len(dataset) > 0

    original, input_patch, mask = dataset[0]
    assert original.shape == input_patch.shape
    assert original.shape[0] == 1  # channel dim
    assert mask.shape == original.shape


def test_grid_masked_dataset_2d():
    """GridMaskedDataset works with 2D images."""
    image = numpy.random.rand(1, 1, 32, 32).astype(numpy.float32)
    dataset = GridMaskedDataset(image)

    assert len(dataset) == 4

    for i in range(4):
        original, input_patch, mask = dataset[i]
        assert original.shape == input_patch.shape
        assert original.shape[0] == 1  # channel dim
        assert mask.shape == original.shape


def test_grid_masked_dataset_3d():
    """GridMaskedDataset works with 3D images (regression test for conv2d bug)."""
    image = numpy.random.rand(1, 1, 8, 8, 8).astype(numpy.float32)
    dataset = GridMaskedDataset(image)

    assert len(dataset) == 4

    for i in range(4):
        original, input_patch, mask = dataset[i]
        assert original.shape == input_patch.shape
        assert original.shape[0] == 1  # channel dim
        assert mask.shape == original.shape


def test_interpolate_mask_preserves_unmasked_2d():
    """Unmasked pixels retain their original values after interpolation (2D)."""
    image = numpy.random.rand(1, 1, 32, 32).astype(numpy.float32)
    dataset = GridMaskedDataset(image)

    original, input_patch, mask = dataset[0]

    # Where mask is 0 (unmasked), the input should equal the original
    unmasked = mask == 0
    assert torch.allclose(input_patch[unmasked], original[unmasked])


def test_interpolate_mask_preserves_unmasked_3d():
    """Unmasked pixels retain their original values after interpolation (3D)."""
    image = numpy.random.rand(1, 1, 8, 8, 8).astype(numpy.float32)
    dataset = GridMaskedDataset(image)

    original, input_patch, mask = dataset[0]

    # Where mask is 0 (unmasked), the input should equal the original
    unmasked = mask == 0
    assert torch.allclose(input_patch[unmasked], original[unmasked])
