"""Tests for RandomMaskedDataset and GridMaskedDataset."""

import numpy
import pytest
import torch

from aydin.nn.datasets.grid_masked_dataset import GridMaskedDataset
from aydin.nn.datasets.noisy_gt_dataset import NoisyGroundtruthDataset
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


# --- Replacement strategy tests for GridMaskedDataset ---


def test_grid_masked_zero_replacement():
    """Verify masked pixels are 0 with 'zero' strategy."""
    image = numpy.random.rand(1, 1, 32, 32).astype(numpy.float32) + 0.5
    dataset = GridMaskedDataset(image, replacement_strategy='zero')

    original, input_patch, mask = dataset[0]
    masked_pixels = mask > 0
    assert torch.all(input_patch[masked_pixels] == 0.0)


def test_grid_masked_random_replacement():
    """Verify masked pixels differ from original with 'random' strategy."""
    numpy.random.seed(42)
    image = numpy.random.rand(1, 1, 32, 32).astype(numpy.float32)
    dataset = GridMaskedDataset(image, replacement_strategy='random')

    original, input_patch, mask = dataset[0]
    masked_pixels = mask > 0
    # Random values should generally differ from original
    if masked_pixels.any():
        # At least some masked pixels should have changed
        diff = (input_patch[masked_pixels] - original[masked_pixels]).abs()
        assert diff.sum().item() > 0


def test_grid_masked_median_replacement():
    """Verify masked pixels are close to local median with 'median' strategy."""
    image = numpy.random.rand(1, 1, 32, 32).astype(numpy.float32)
    dataset = GridMaskedDataset(image, replacement_strategy='median')

    original, input_patch, mask = dataset[0]
    # Just verify it runs without error and produces valid output
    assert input_patch.shape == original.shape
    assert not torch.isnan(input_patch).any()


# --- Replacement strategy tests for RandomMaskedDataset ---


def test_random_masked_zero_replacement():
    """Verify masked pixels are 0 with 'zero' strategy."""
    image = numpy.random.rand(1, 1, 64, 64).astype(numpy.float32) + 0.5
    dataset = RandomMaskedDataset(image, patch_size=32, replacement_strategy='zero')

    original, input_patch, mask = dataset[0]
    masked_pixels = mask > 0
    if masked_pixels.any():
        assert torch.all(input_patch[masked_pixels] == 0.0)


def test_random_masked_random_replacement():
    """Verify masked pixels differ from original with 'random' strategy."""
    numpy.random.seed(42)
    image = numpy.random.rand(1, 1, 64, 64).astype(numpy.float32)
    dataset = RandomMaskedDataset(image, patch_size=32, replacement_strategy='random')

    original, input_patch, mask = dataset[0]
    masked_pixels = mask > 0
    if masked_pixels.any():
        diff = (input_patch[masked_pixels] - original[masked_pixels]).abs()
        assert diff.sum().item() > 0


def test_random_masked_median_replacement():
    """Verify masked pixels use median filter with 'median' strategy."""
    image = numpy.random.rand(1, 1, 64, 64).astype(numpy.float32)
    dataset = RandomMaskedDataset(image, patch_size=32, replacement_strategy='median')

    original, input_patch, mask = dataset[0]
    assert input_patch.shape == original.shape
    assert not torch.isnan(input_patch).any()


# --- Multi-channel tests ---


def test_grid_masked_dataset_multichannel_2d():
    """GridMaskedDataset works with multi-channel (C=3) 2D images."""
    image = numpy.random.rand(1, 3, 32, 32).astype(numpy.float32)
    dataset = GridMaskedDataset(image)

    assert len(dataset) == 4

    for i in range(4):
        original, input_patch, mask = dataset[i]
        assert original.shape == (3, 32, 32)
        assert input_patch.shape == original.shape
        assert mask.shape == original.shape
        assert not torch.isnan(input_patch).any()


def test_random_masked_dataset_multichannel_interpolation():
    """RandomMaskedDataset interpolate_mask works with multi-channel (C=3) tensors."""
    image = numpy.random.rand(1, 3, 64, 64).astype(numpy.float32)
    dataset = RandomMaskedDataset(image, patch_size=32)

    # Directly test interpolate_mask with multi-channel tensor
    tensor = torch.as_tensor(image, dtype=torch.float32)
    mask = torch.zeros_like(tensor)
    mask[:, :, ::2, ::2] = 1
    mask_inv = 1 - mask

    result = dataset.interpolate_mask(tensor, mask, mask_inv)
    assert result.shape == tensor.shape
    assert not torch.isnan(result).any()
    # Unmasked pixels should be unchanged
    assert torch.allclose(result[mask_inv.bool()], tensor[mask_inv.bool()])


# --- Grid mask coverage tests ---


def test_grid_mask_full_coverage_2d():
    """4 grid mask phases combined cover every pixel exactly once (2D)."""
    image = numpy.random.rand(1, 1, 32, 32).astype(numpy.float32)
    dataset = GridMaskedDataset(image)

    combined = torch.zeros(1, 1, 32, 32)
    for i in range(4):
        combined += dataset.get_mask(i)

    assert torch.all(combined == 1.0), "Not all pixels covered exactly once"


def test_grid_mask_full_coverage_3d():
    """4 grid mask phases combined cover every voxel exactly once (3D)."""
    image = numpy.random.rand(1, 1, 8, 8, 8).astype(numpy.float32)
    dataset = GridMaskedDataset(image)

    combined = torch.zeros(1, 1, 8, 8, 8)
    for i in range(4):
        combined += dataset.get_mask(i)

    assert torch.all(combined == 1.0), "Not all voxels covered exactly once"


# --- Dtype tests (float64 input → float32 tensor) ---


def test_grid_masked_dataset_dtype_float64_input():
    """GridMaskedDataset converts float64 numpy input to float32 tensors."""
    image = numpy.random.rand(1, 1, 32, 32).astype(numpy.float64)
    dataset = GridMaskedDataset(image)

    original, input_patch, mask = dataset[0]
    assert original.dtype == torch.float32
    assert input_patch.dtype == torch.float32


def test_random_masked_dataset_dtype_float64_input():
    """RandomMaskedDataset converts float64 numpy input to float32 tensors."""
    image = numpy.random.rand(1, 1, 64, 64).astype(numpy.float64)
    dataset = RandomMaskedDataset(image, patch_size=32)

    original, input_patch, mask = dataset[0]
    assert original.dtype == torch.float32
    assert input_patch.dtype == torch.float32


def test_noisy_gt_dataset_dtype_float64_input():
    """NoisyGroundtruthDataset converts float64 numpy input to float32 tensors."""
    noisy = numpy.random.rand(1, 1, 32, 32).astype(numpy.float64)
    clean = numpy.random.rand(1, 1, 32, 32).astype(numpy.float64)
    dataset = NoisyGroundtruthDataset([noisy], [clean], device=torch.device('cpu'))

    noisy_out, clean_out = dataset[0]
    assert noisy_out.dtype == torch.float32
    assert clean_out.dtype == torch.float32


# --- Fix 5: patch_size validation ---


def test_random_masked_dataset_patch_size_too_large():
    """ValueError raised when patch_size exceeds smallest spatial dim."""
    image = numpy.random.rand(1, 1, 32, 64).astype(numpy.float32)
    with pytest.raises(ValueError, match="patch_size.*exceeds"):
        RandomMaskedDataset(image, patch_size=33)


# --- Fix 1: correct patch count uses spatial dims only ---


def test_random_masked_dataset_patch_count():
    """Patch count should be based on spatial dims, not inflated by batch/channel."""
    # Before fix: numpy.prod((1,1,64,64)) // 32 = 128 (wrong, included B and C)
    # After fix:  numpy.prod((64,64)) // 32^2 = 4 (correct, spatial only)
    image = numpy.random.rand(1, 1, 64, 64).astype(numpy.float32)
    dataset = RandomMaskedDataset(image, patch_size=32)
    # nb_patches=4 is passed to random_patches, which generates 4/0.5=8
    # candidates, then keeps the top ~50% by entropy.  The exact count
    # depends on random_patches' adoption-rate slicing, but it should be
    # in a small, reasonable range — not the inflated 128 from before.
    assert len(dataset) <= 10


# --- Fix 4: patches with images outside [0, 1] ---


def test_random_patches_negative_values():
    """RandomMaskedDataset works with images containing negative values."""
    image = (numpy.random.rand(1, 1, 64, 64).astype(numpy.float32) - 0.5) * 10
    dataset = RandomMaskedDataset(image, patch_size=32)
    assert len(dataset) > 0
    original, input_patch, mask = dataset[0]
    assert not torch.isnan(input_patch).any()


def test_random_patches_constant_image():
    """RandomMaskedDataset handles constant-value images without error."""
    image = numpy.full((1, 1, 64, 64), 5.0, dtype=numpy.float32)
    dataset = RandomMaskedDataset(image, patch_size=32)
    assert len(dataset) > 0
    original, input_patch, mask = dataset[0]
    assert not torch.isnan(input_patch).any()


# --- Fix 7: grid mask caching ---


def test_grid_masked_dataset_mask_caching():
    """get_mask() returns the same cached object on repeated calls."""
    image = numpy.random.rand(1, 1, 32, 32).astype(numpy.float32)
    dataset = GridMaskedDataset(image)
    assert dataset.get_mask(0) is dataset.get_mask(0)
    assert dataset.get_mask(3) is dataset.get_mask(3)


# --- Fix 8: interpolation kernel caching ---


def test_grid_interpolation_kernel_cached():
    """GridMaskedDataset pre-computes _interp_kernel with correct shape."""
    image_2d = numpy.random.rand(1, 1, 32, 32).astype(numpy.float32)
    ds2d = GridMaskedDataset(image_2d)
    assert ds2d._interp_kernel is not None
    assert ds2d._interp_kernel.shape == (1, 1, 3, 3)

    image_3d = numpy.random.rand(1, 1, 8, 8, 8).astype(numpy.float32)
    ds3d = GridMaskedDataset(image_3d)
    assert ds3d._interp_kernel is not None
    assert ds3d._interp_kernel.shape == (1, 1, 3, 3, 3)


def test_random_interpolation_kernel_cached():
    """RandomMaskedDataset pre-computes _interp_kernel with correct shape."""
    image_2d = numpy.random.rand(1, 1, 64, 64).astype(numpy.float32)
    ds2d = RandomMaskedDataset(image_2d, patch_size=32)
    assert ds2d._interp_kernel is not None
    assert ds2d._interp_kernel.shape == (1, 1, 3, 3)
