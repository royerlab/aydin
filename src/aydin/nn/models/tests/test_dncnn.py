"""Forward pass tests for the DnCNN model architecture."""

import pytest
import torch

from aydin.nn.models.dncnn import DnCNNModel


@pytest.mark.parametrize(
    "input_array_shape, spacetime_ndim",
    [
        ((1, 1, 64, 64), 2),
        ((1, 1, 16, 16, 16), 3),
    ],
)
def test_forward(input_array_shape, spacetime_ndim):
    """Test that DnCNN forward pass preserves input shape and dtype."""
    input_array = torch.randn(input_array_shape)
    model = DnCNNModel(spacetime_ndim=spacetime_ndim, num_of_layers=5)
    result = model(input_array)

    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
    assert torch.isfinite(result).all()
    assert not torch.allclose(result, torch.zeros_like(result))


@pytest.mark.parametrize(
    "input_array_shape, spacetime_ndim, num_of_layers",
    [
        ((1, 1, 32, 32), 2, 3),
        ((1, 1, 32, 32), 2, 9),
        ((1, 1, 32, 32), 2, 17),
        ((1, 1, 16, 16, 16), 3, 5),
    ],
)
def test_various_depths(input_array_shape, spacetime_ndim, num_of_layers):
    """Test DnCNN forward pass with different network depths."""
    input_array = torch.randn(input_array_shape)
    model = DnCNNModel(spacetime_ndim=spacetime_ndim, num_of_layers=num_of_layers)
    result = model(input_array)

    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
    assert torch.isfinite(result).all()
    assert not torch.allclose(result, torch.zeros_like(result))


def test_invalid_spacetime_ndim():
    """Test that DnCNN raises ValueError for invalid spacetime_ndim."""
    with pytest.raises(ValueError, match="spacetime_ndim must be 2 or 3"):
        DnCNNModel(spacetime_ndim=4)


def test_gradient_flow():
    """Verify gradients flow through the entire DnCNN."""
    input_array = torch.randn(1, 1, 32, 32, requires_grad=True)
    model = DnCNNModel(spacetime_ndim=2, num_of_layers=5)
    result = model(input_array)
    loss = result.sum()
    loss.backward()
    assert input_array.grad is not None
    assert input_array.grad.shape == input_array.shape
    assert not torch.allclose(input_array.grad, torch.zeros_like(input_array.grad))
