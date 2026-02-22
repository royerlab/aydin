"""Tests for CustomConv layer and double_conv_block."""

import torch

from aydin.nn.layers.custom_conv import CustomConv, double_conv_block


def test_custom_conv_2d_shape():
    """2D convolution should preserve spatial dimensions."""
    layer = CustomConv(1, 16, spacetime_ndim=2, kernel_size=3)
    x = torch.randn(1, 1, 32, 32)
    y = layer(x)
    assert y.shape == (1, 16, 32, 32)


def test_custom_conv_3d_shape():
    """3D convolution should preserve spatial dimensions."""
    layer = CustomConv(1, 8, spacetime_ndim=3, kernel_size=3)
    x = torch.randn(1, 1, 16, 16, 16)
    y = layer(x)
    assert y.shape == (1, 8, 16, 16, 16)


def test_custom_conv_reflect_2d():
    """Reflect padding mode should preserve spatial dimensions in 2D."""
    layer = CustomConv(1, 8, spacetime_ndim=2, kernel_size=3, padding_mode='reflect')
    x = torch.randn(1, 1, 32, 32)
    y = layer(x)
    assert y.shape == (1, 8, 32, 32)


def test_custom_conv_reflect_3d():
    """Reflect padding mode should preserve spatial dimensions in 3D."""
    layer = CustomConv(1, 8, spacetime_ndim=3, kernel_size=3, padding_mode='reflect')
    x = torch.randn(1, 1, 16, 16, 16)
    y = layer(x)
    assert y.shape == (1, 8, 16, 16, 16)


def test_custom_conv_instance_norm():
    """Instance normalization should work without error."""
    layer = CustomConv(1, 8, spacetime_ndim=2, normalization='instance')
    x = torch.randn(2, 1, 32, 32)
    y = layer(x)
    assert y.shape == (2, 8, 32, 32)


def test_custom_conv_batch_norm():
    """Batch normalization should work without error."""
    layer = CustomConv(1, 8, spacetime_ndim=2, normalization='batch')
    x = torch.randn(2, 1, 32, 32)
    y = layer(x)
    assert y.shape == (2, 8, 32, 32)


def test_double_conv_block():
    """double_conv_block should produce correct output shape."""
    block = double_conv_block(1, 16, 32, spacetime_ndim=2)
    x = torch.randn(1, 1, 32, 32)
    y = block(x)
    assert y.shape == (1, 32, 32, 32)
