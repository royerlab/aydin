"""Tests for DilatedConv layer."""

import pytest
import torch

from aydin.nn.layers.dilated_conv import DilatedConv


def test_dilated_conv_2d():
    """2D dilated convolution should produce correct output shape."""
    layer = DilatedConv(1, 8, spacetime_ndim=2, padding=2, kernel_size=3, dilation=2)
    x = torch.randn(1, 1, 32, 32)
    y = layer(x)
    assert y.shape[0] == 1
    assert y.shape[1] == 8


def test_dilated_conv_3d():
    """3D dilated convolution should produce correct output shape."""
    layer = DilatedConv(1, 8, spacetime_ndim=3, padding=1, kernel_size=3, dilation=1)
    x = torch.randn(1, 1, 16, 16, 16)
    y = layer(x)
    assert y.shape[0] == 1
    assert y.shape[1] == 8


def test_dilated_conv_invalid_ndim():
    """Invalid spacetime_ndim should raise ValueError."""
    with pytest.raises(ValueError, match="spacetime_ndim"):
        DilatedConv(1, 8, spacetime_ndim=4, padding=1, kernel_size=3, dilation=1)


def test_dilated_conv_activations():
    """Different activation functions should all work."""
    for act in ['ReLU', 'swish', 'lrel']:
        layer = DilatedConv(
            1, 4, spacetime_ndim=2, padding=1, kernel_size=3, dilation=1, activation=act
        )
        x = torch.randn(1, 1, 16, 16)
        y = layer(x)
        assert y.shape[1] == 4
