"""Tests for PyTorch device utilities."""

from unittest.mock import MagicMock, patch

import torch

from aydin.util.torch.device import (
    available_device_memory,
    clear_device_cache,
    get_torch_device,
)


def test_get_torch_device_returns_device():
    """Test that get_torch_device returns a valid torch.device."""
    device = get_torch_device()
    assert isinstance(device, torch.device)
    assert device.type in ('cpu', 'cuda', 'mps')


def test_get_torch_device_consistent():
    """Test that repeated calls return the same device."""
    device1 = get_torch_device()
    device2 = get_torch_device()
    assert device1 == device2


def test_available_device_memory_positive():
    """Test that available_device_memory returns a positive value."""
    memory = available_device_memory()
    assert memory > 0
    assert isinstance(memory, (int, float))


def test_get_torch_device_cpu_path():
    """When CUDA and MPS are unavailable, should return CPU device."""
    with patch('aydin.util.torch.device.torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.device = torch.device
        device = get_torch_device()
        assert str(device) == 'cpu'


def test_get_torch_device_cuda_path():
    """When CUDA is available, should return cuda:0 device."""
    with patch('aydin.util.torch.device.torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device = torch.device
        device = get_torch_device()
        assert 'cuda' in str(device)


def test_get_torch_device_mps_path():
    """When CUDA is unavailable but MPS is, should return mps device."""
    with patch('aydin.util.torch.device.torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.device = torch.device
        device = get_torch_device()
        assert str(device) == 'mps'


def test_available_device_memory_cpu_path():
    """When CUDA and MPS are unavailable, should return full system RAM."""
    mock_vm = MagicMock()
    mock_vm.available = 8 * 1024**3
    mock_psutil = MagicMock()
    mock_psutil.virtual_memory.return_value = mock_vm
    with patch('aydin.util.torch.device.torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict('sys.modules', {'psutil': mock_psutil}):
            mem = available_device_memory()
            assert mem == float(8 * 1024**3)


def test_available_device_memory_mps_path():
    """When MPS is available (no CUDA), should return 40% of system RAM."""
    mock_vm = MagicMock()
    mock_vm.available = 16 * 1024**3
    mock_psutil = MagicMock()
    mock_psutil.virtual_memory.return_value = mock_vm
    with patch('aydin.util.torch.device.torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict('sys.modules', {'psutil': mock_psutil}):
            mem = available_device_memory()
            # MPS shares unified memory, so 40% of system RAM is returned
            assert mem == float(16 * 1024**3) * 0.4


def test_available_device_memory_cuda_path():
    """When CUDA is available, should return GPU free memory."""
    with patch('aydin.util.torch.device.torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (4 * 1024**3, 12 * 1024**3)
        mem = available_device_memory()
        assert mem == float(4 * 1024**3)


def test_clear_device_cache_no_raise():
    """clear_device_cache should not raise on any platform."""
    clear_device_cache()


def test_clear_device_cache_mps_calls_synchronize():
    """On MPS, synchronize must be called before empty_cache."""
    call_order = []
    with patch('aydin.util.torch.device.torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.mps.synchronize = MagicMock(
            side_effect=lambda: call_order.append('synchronize')
        )
        mock_torch.mps.empty_cache = MagicMock(
            side_effect=lambda: call_order.append('empty_cache')
        )
        clear_device_cache()
        assert call_order == ['synchronize', 'empty_cache']
