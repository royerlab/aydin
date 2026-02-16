"""Post-training center pixel smoothing for JINet models.

After self-supervised training with a blind-spot constraint, the center
pixel weight in the first dilated convolution layer is zero. This module
applies a smoothing convolution to the kernel weights and re-introduces
a small amount of center pixel information, improving inference quality.
"""

import numpy as np
import torch
import torch.nn.functional as F

from aydin.nn.layers.dilated_conv import DilatedConv
from aydin.util.log.log import aprint


def apply_center_smoothing(model, spacetime_ndim=2, blind_spots=None):
    """Apply center-pixel smoothing to the first DilatedConv layer.

    After self-supervised blind-spot training, the center pixel weight
    in dilated convolution kernels is zero. This function applies a
    smoothing convolution to the spatial kernel weights of the first
    ``DilatedConv`` layer, copies the smoothed center pixel value back,
    and rescales the weights to preserve the total sum.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model containing ``DilatedConv`` layers (e.g. JINet).
    spacetime_ndim : int
        Number of spatial dimensions (2 or 3).
    blind_spots : list of tuple of int, optional
        Additional blind-spot positions relative to center. Each tuple
        has length ``spacetime_ndim``. These positions are zeroed in
        the smoothing kernel.

    Returns
    -------
    bool
        ``True`` if smoothing was applied, ``False`` if no ``DilatedConv``
        layer was found.
    """
    # Find the first DilatedConv layer
    target_layer = None
    for module in model.modules():
        if isinstance(module, DilatedConv):
            target_layer = module
            break

    if target_layer is None:
        aprint("No DilatedConv layer found, skipping center smoothing.")
        return False

    # Get the convolution weight: shape (out_ch, in_ch, *spatial)
    weight = target_layer.conv.weight.data

    if spacetime_ndim == 2:
        _smooth_2d(weight, blind_spots)
    elif spacetime_ndim == 3:
        _smooth_3d(weight, blind_spots)
    else:
        raise ValueError(f"Unsupported spacetime_ndim: {spacetime_ndim}")

    aprint("Center smoothing applied to first DilatedConv layer.")
    return True


def _smooth_2d(weight, blind_spots=None):
    """Apply 2D center smoothing to conv weight tensor.

    Parameters
    ----------
    weight : torch.Tensor
        Convolution weight tensor of shape ``(out_ch, in_ch, H, W)``.
    blind_spots : list of tuple of int, optional
        Blind-spot positions relative to the center.
    """
    out_ch, in_ch, kh, kw = weight.shape
    center_h, center_w = kh // 2, kw // 2

    # Create smoothing kernel
    kernel = np.ones((3, 3), dtype=np.float32)

    # Zero out blind spot positions in the kernel
    if blind_spots:
        for spot in blind_spots:
            shifted = (spot[0] + 1, spot[1] + 1)
            if 0 <= shifted[0] < 3 and 0 <= shifted[1] < 3:
                kernel[shifted] = 0

    kernel /= kernel.sum()

    # For each (out_ch, in_ch) slice, smooth the spatial kernel
    # and extract the smoothed center pixel value
    smoothing_kernel = torch.as_tensor(
        kernel[np.newaxis, np.newaxis, :, :], device=weight.device
    )

    for oc in range(out_ch):
        original_sum = weight[oc].sum().item()

        for ic in range(in_ch):
            # Extract the spatial kernel slice: (H, W)
            w_slice = weight[oc, ic].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            # Apply smoothing convolution with 'same' padding
            smoothed = F.conv2d(w_slice, smoothing_kernel, padding=1)

            # Get the smoothed center value
            smoothed_center = smoothed[0, 0, center_h, center_w].item()

            # Copy smoothed center value back
            weight[oc, ic, center_h, center_w] = smoothed_center

        # Rescale per output channel to preserve its weight sum
        new_sum = weight[oc].sum().item()
        if abs(new_sum) > 1e-10:
            weight[oc] *= original_sum / new_sum


def _smooth_3d(weight, blind_spots=None):
    """Apply 3D center smoothing to conv weight tensor.

    Parameters
    ----------
    weight : torch.Tensor
        Convolution weight tensor of shape ``(out_ch, in_ch, D, H, W)``.
    blind_spots : list of tuple of int, optional
        Blind-spot positions relative to the center.
    """
    out_ch, in_ch, kd, kh, kw = weight.shape
    center_d, center_h, center_w = kd // 2, kh // 2, kw // 2

    # Create 3D smoothing kernel
    kernel = np.ones((3, 3, 3), dtype=np.float32)

    if blind_spots:
        for spot in blind_spots:
            shifted = (spot[0] + 1, spot[1] + 1, spot[2] + 1)
            if all(0 <= s < 3 for s in shifted):
                kernel[shifted] = 0

    kernel /= kernel.sum()

    smoothing_kernel = torch.as_tensor(
        kernel[np.newaxis, np.newaxis, :, :, :], device=weight.device
    )

    for oc in range(out_ch):
        original_sum = weight[oc].sum().item()

        for ic in range(in_ch):
            w_slice = weight[oc, ic].unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            smoothed = F.conv3d(w_slice, smoothing_kernel, padding=1)
            smoothed_center = smoothed[0, 0, center_d, center_h, center_w].item()
            weight[oc, ic, center_d, center_h, center_w] = smoothed_center

        # Rescale per output channel to preserve its weight sum
        new_sum = weight[oc].sum().item()
        if abs(new_sum) > 1e-10:
            weight[oc] *= original_sum / new_sum
