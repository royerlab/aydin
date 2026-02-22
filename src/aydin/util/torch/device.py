"""PyTorch device selection utility."""

import os

import torch

from aydin.util.log.log import aprint


def get_torch_device():
    """Select the best available PyTorch device.

    Returns a CUDA device if a GPU is available, an MPS device on Apple
    Silicon Macs, or falls back to CPU.  When MPS is selected, sets
    ``PYTORCH_ENABLE_MPS_FALLBACK=1`` so that unsupported operations
    fall back to CPU instead of raising an error.

    Returns
    -------
    torch.device
        The selected device (``cuda:0``, ``mps``, or ``cpu``).
    """
    if torch.cuda.is_available():
        dev = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    else:
        dev = "cpu"
    device = torch.device(dev)
    aprint(f"device {device}")
    return device


def available_device_memory():
    """Return the available (free) memory of the best available device in bytes.

    For CUDA devices, returns the free GPU memory. For MPS (Apple Silicon),
    returns 40% of available system memory — MPS shares unified memory with
    the OS, CPU-side numpy arrays, and the PyTorch runtime, so using 100%
    leads to OOM. For CPU, returns the full available system memory.

    Returns
    -------
    float
        Available device memory in bytes.
    """
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return float(free)
    try:
        import psutil

        available = float(psutil.virtual_memory().available)
    except ImportError:
        aprint(
            "Warning: psutil is not installed — cannot determine available CPU memory. "
            "Install it with: pip install psutil"
        )
        return 8e9

    # MPS shares unified memory with OS and CPU-side allocations
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return available * 0.4

    return available


def clear_device_cache():
    """Release cached memory on the active PyTorch device.

    For CUDA, calls ``torch.cuda.empty_cache()``.
    For MPS, calls ``torch.mps.synchronize()`` then ``torch.mps.empty_cache()``.
    The synchronize is essential: without it, async MPS operations still hold
    memory references, causing accumulation across batch items.
    For CPU, this is a no-op.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
