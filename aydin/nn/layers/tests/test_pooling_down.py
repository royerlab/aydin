"""Tests for PoolingDown layer."""

import pytest
import torch

from aydin.nn.layers.pooling_down import PoolingDown


def test_avg_pool_2d():
    """Average pooling should halve spatial dimensions in 2D."""
    layer = PoolingDown(spacetime_ndim=2, pooling_mode='ave')
    x = torch.randn(1, 4, 32, 32)
    y = layer(x)
    assert y.shape == (1, 4, 16, 16)


def test_max_pool_2d():
    """Max pooling should halve spatial dimensions in 2D."""
    layer = PoolingDown(spacetime_ndim=2, pooling_mode='max')
    x = torch.randn(1, 4, 32, 32)
    y = layer(x)
    assert y.shape == (1, 4, 16, 16)


def test_pool_3d():
    """Pooling should halve spatial dimensions in 3D."""
    layer = PoolingDown(spacetime_ndim=3, pooling_mode='ave')
    x = torch.randn(1, 4, 16, 16, 16)
    y = layer(x)
    assert y.shape == (1, 4, 8, 8, 8)


def test_invalid_mode():
    """Invalid pooling mode should raise ValueError at construction."""
    with pytest.raises(ValueError, match='pooling_mode must be'):
        PoolingDown(spacetime_ndim=2, pooling_mode='invalid')
