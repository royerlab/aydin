"""PyTorch device selection utility."""

import psutil
import torch

from aydin.util.log.log import aprint


def get_torch_device():
    """Select the best available PyTorch device.

    Returns a CUDA device if a GPU is available, otherwise falls back
    to CPU.

    Returns
    -------
    torch.device
        The selected device (``cuda:0`` or ``cpu``).
    """
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    aprint(f"device {device}")
    return device


def available_device_memory():
    """Return the available (free) memory of the best available device in bytes.

    For CUDA devices, returns the free GPU memory. For CPU, returns
    the available system memory.

    Returns
    -------
    float
        Available device memory in bytes.
    """
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return float(free)
    return float(psutil.virtual_memory().available)
