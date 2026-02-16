"""Tests for the PyTorch UNet model forward pass.

Covers standard 2D/3D shapes, deep UNet levels, non-standard 3D shapes
(various depth dimensions), and thin volume edge cases. The non-standard
shape tests are migrated from the former TF test_various_masking_3D and
test_thin_masking_3D tests.
"""

import pytest
import torch

from aydin.nn.models.unet import UNetModel


@pytest.mark.parametrize(
    "input_array_shape, spacetime_ndim, nb_unet_levels",
    [
        ((1, 1, 64, 64), 2, 2),
        ((1, 1, 64, 64), 2, 3),
        ((1, 1, 64, 64), 2, 4),
        ((1, 1, 32, 32, 32), 3, 2),
        ((1, 1, 32, 32, 32), 3, 3),
    ],
)
def test_forward(input_array_shape, nb_unet_levels, spacetime_ndim):
    """Test that UNet forward pass preserves input shape and dtype."""
    input_array = torch.randn(input_array_shape)
    model = UNetModel(
        nb_unet_levels=nb_unet_levels,
        spacetime_ndim=spacetime_ndim,
    )
    result = model(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
    assert torch.isfinite(result).all()
    assert not torch.allclose(result, torch.zeros_like(result))


@pytest.mark.parametrize(
    "depth",
    [21, 25],
)
def test_various_3d_depth(depth):
    """Verify UNet handles various non-standard 3D depth dimensions.

    Migrated from TF test_various_masking_3D -- tests that the model
    correctly handles depths that are not powers of 2.
    """
    input_array = torch.randn(1, 1, depth, 64, 64)
    model = UNetModel(nb_unet_levels=4, spacetime_ndim=3)
    result = model(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
    # Verify model produces non-trivial output
    assert not torch.allclose(result, torch.zeros_like(result))


@pytest.mark.parametrize(
    "depth, nb_levels",
    [
        (4, 1),  # thin: 4 -> 2 -> upsample back to 4
        (8, 2),  # thin: 8 -> 4 -> 2 -> upsample
        (16, 3),  # moderate: 16 -> 8 -> 4 -> 2 -> upsample
    ],
)
def test_thin_volume_3d(depth, nb_levels):
    """Verify UNet handles thin 3D volumes with appropriate level count.

    Migrated from TF test_thin_masking_3D. Note: PyTorch's max_pool3d
    requires each spatial dim >= 2 after pooling, so nb_unet_levels must
    be chosen so that depth / 2^levels >= 1.
    """
    input_array = torch.randn(1, 1, depth, 64, 64)
    model = UNetModel(nb_unet_levels=nb_levels, spacetime_ndim=3)
    result = model(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
    # Verify model produces non-trivial output
    assert not torch.allclose(result, torch.zeros_like(result))


def test_non_power_of_2_2d():
    """Verify UNet handles non-power-of-2 2D spatial dimensions."""
    input_array = torch.randn(1, 1, 100, 100)
    model = UNetModel(nb_unet_levels=3, spacetime_ndim=2)
    result = model(input_array)
    assert result.shape == input_array.shape


def test_batch_gt_one():
    """Verify UNet handles batch sizes greater than 1."""
    input_array = torch.randn(4, 1, 32, 32)
    model = UNetModel(nb_unet_levels=2, spacetime_ndim=2)
    result = model(input_array)
    assert result.shape == input_array.shape
    assert torch.isfinite(result).all()


def test_average_pooling():
    """Verify UNet works with average pooling mode."""
    input_array = torch.randn(1, 1, 64, 64)
    model = UNetModel(nb_unet_levels=3, spacetime_ndim=2, pooling_mode='ave')
    result = model(input_array)
    assert result.shape == input_array.shape
    assert torch.isfinite(result).all()
    assert not torch.allclose(result, torch.zeros_like(result))


def test_odd_dimensions():
    """Verify UNet handles odd spatial dimensions correctly."""
    input_array = torch.randn(1, 1, 33, 47)
    model = UNetModel(nb_unet_levels=3, spacetime_ndim=2)
    result = model(input_array)
    assert result.shape == input_array.shape
    assert torch.isfinite(result).all()


def test_gradient_flow():
    """Verify gradients flow through the entire UNet."""
    input_array = torch.randn(1, 1, 32, 32, requires_grad=True)
    model = UNetModel(nb_unet_levels=2, spacetime_ndim=2)
    result = model(input_array)
    loss = result.sum()
    loss.backward()
    assert input_array.grad is not None
    assert input_array.grad.shape == input_array.shape
    assert not torch.allclose(input_array.grad, torch.zeros_like(input_array.grad))
