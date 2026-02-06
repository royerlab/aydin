"""PyTorch device selection utility."""

import torch

from aydin.util.log.log import lprint


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
    lprint(f"device {device}")
    return device
