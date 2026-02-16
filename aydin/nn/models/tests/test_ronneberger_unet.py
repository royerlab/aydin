"""Forward pass tests for the Ronneberger UNet model architecture."""

import pytest
import torch

from aydin.nn.models.ronneberger_unet import RonnebergerUNetModel


@pytest.mark.parametrize(
    "input_array_shape, spacetime_ndim, depth",
    [
        ((1, 1, 64, 64), 2, 2),
        ((1, 1, 64, 64), 2, 3),
        ((1, 1, 128, 128), 2, 4),
        ((1, 1, 32, 32, 32), 3, 2),
        ((1, 1, 32, 32, 32), 3, 3),
    ],
)
def test_forward(input_array_shape, spacetime_ndim, depth):
    """Test that Ronneberger UNet forward pass preserves input shape and dtype."""
    input_array = torch.zeros(input_array_shape)
    model = RonnebergerUNetModel(spacetime_ndim=spacetime_ndim, depth=depth)
    result = model(input_array)

    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


@pytest.mark.parametrize(
    "up_mode",
    ['upconv', 'upsample'],
)
def test_upsampling_modes(up_mode):
    """Test Ronneberger UNet with different upsampling modes."""
    input_array = torch.zeros((1, 1, 64, 64))
    model = RonnebergerUNetModel(spacetime_ndim=2, depth=2, up_mode=up_mode)
    result = model(input_array)

    assert result.shape == input_array.shape


def test_with_batch_norm():
    """Test Ronneberger UNet with batch normalization enabled."""
    input_array = torch.zeros((1, 1, 64, 64))
    model = RonnebergerUNetModel(spacetime_ndim=2, depth=2, batch_norm=True)
    result = model(input_array)

    assert result.shape == input_array.shape


@pytest.mark.parametrize(
    "input_array_shape, spacetime_ndim, depth",
    [
        ((1, 1, 65, 65), 2, 2),
        ((1, 1, 33, 33, 33), 3, 2),
    ],
)
def test_forward_odd_dimensions(input_array_shape, spacetime_ndim, depth):
    """Test that Ronneberger UNet handles odd spatial dimensions without crashing."""
    input_array = torch.zeros(input_array_shape)
    model = RonnebergerUNetModel(spacetime_ndim=spacetime_ndim, depth=depth)
    result = model(input_array)

    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
