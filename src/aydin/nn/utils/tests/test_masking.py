"""Tests for the random pixel masking wrapper."""

import torch
from torch import nn

from aydin.nn.utils.masking import Masking


def test_output_shape():
    """Masking wrapper should preserve output shape."""
    model = nn.Conv2d(1, 1, 3, padding=1)
    wrapped = Masking(model, density=0.1)
    x = torch.randn(1, 1, 32, 32)
    result = wrapped(x)
    assert result.shape == x.shape


def test_mask_shape():
    """Mask should have the same shape as input."""
    model = nn.Conv2d(1, 1, 3, padding=1)
    wrapped = Masking(model, density=0.1)
    x = torch.randn(1, 1, 32, 32)
    wrapped(x)
    mask = wrapped.get_mask()
    assert mask.shape == x.shape


def test_mask_density():
    """Mask density should be approximately the configured value."""
    model = nn.Conv2d(1, 1, 3, padding=1)
    wrapped = Masking(model, density=0.1)
    x = torch.randn(1, 1, 256, 256)
    wrapped(x)
    mask = wrapped.get_mask()
    actual_density = mask.float().mean().item()
    assert abs(actual_density - 0.1) < 0.02


def test_mask_is_boolean():
    """Mask should be a boolean tensor."""
    model = nn.Conv2d(1, 1, 3, padding=1)
    wrapped = Masking(model, density=0.1)
    x = torch.randn(1, 1, 32, 32)
    wrapped(x)
    mask = wrapped.get_mask()
    assert mask.dtype == torch.bool


def test_3d_input():
    """Masking should work with 3D inputs."""
    model = nn.Conv3d(1, 1, 3, padding=1)
    wrapped = Masking(model, density=0.2)
    x = torch.randn(1, 1, 16, 16, 16)
    result = wrapped(x)
    assert result.shape == x.shape
    assert wrapped.get_mask().shape == x.shape
