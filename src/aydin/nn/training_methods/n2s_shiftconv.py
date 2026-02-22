"""Noise2Self shift-convolution self-supervised training method for PyTorch.

Implements self-supervised training using the shift-convolution trick:
input images are rotated 4 ways (2D) or 6 ways (3D), concatenated along
the batch dim, and a column/row shift is applied at each convolution to
prevent the network from seeing the center pixel. Outputs are un-rotated
and averaged.
"""

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from aydin.util.log.log import aprint
from aydin.util.torch.device import get_torch_device


def _rotate_2d(x, k):
    """Rotate a 2D image tensor by k * 90 degrees."""
    return torch.rot90(x, k, dims=(-2, -1))


def _shift_right(x):
    """Shift the last spatial dim right by 1 pixel (pad left, crop right).

    This ensures the center pixel is excluded from the receptive field
    when combined with rotations.
    """
    # Pad left by 1, crop right by 1 on the last spatial dimension.
    # F.pad with replicate mode requires:
    #   4D (B,C,H,W): pad size 4 → (left, right, top, bottom)
    #   5D (B,C,D,H,W): pad size 6 → (left, right, top, bottom, front, back)
    if x.ndim == 5:
        x = F.pad(x, (1, 0, 0, 0, 0, 0), mode='replicate')
    else:
        x = F.pad(x, (1, 0, 0, 0), mode='replicate')
    x = x[..., :-1]
    return x


class ShiftConvWrapper(nn.Module):
    """Wrapper that applies shift-convolution around a base UNet model.

    For 2D images, creates 4 rotated copies (0, 90, 180, 270 degrees),
    concatenates them along the batch dimension, applies a column shift
    after the model's forward pass, then splits, un-rotates, and averages
    the outputs.

    Parameters
    ----------
    base_model : nn.Module
        The underlying model (e.g. UNet) to wrap.
    spacetime_ndim : int
        Number of spatial dimensions (2 or 3). Currently only 2D is
        supported.
    """

    def __init__(self, base_model, spacetime_ndim=2):
        """Initialize the shift-convolution wrapper.

        Parameters
        ----------
        base_model : torch.nn.Module
            The underlying model (e.g. UNet) to wrap with the
            shift-convolution self-supervision trick.
        spacetime_ndim : int
            Number of spatial dimensions (2 or 3). Currently only
            2D is fully supported; 3D uses HW-plane rotations.
        """
        super().__init__()
        self.base_model = base_model
        self.spacetime_ndim = spacetime_ndim

        if spacetime_ndim == 2:
            self.n_rotations = 4
        elif spacetime_ndim == 3:
            # Only use HW-plane rotations for 3D to avoid shape mismatches
            # when D != H or D != W
            self.n_rotations = 4
        else:
            raise ValueError("spacetime_ndim must be 2 or 3")

    def forward(self, x):
        """Run shift-convolution forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(B, C, H, W)`` (2D) or
            ``(B, C, D, H, W)`` (3D).

        Returns
        -------
        torch.Tensor
            Denoised output, averaged over all rotation views.
        """
        if self.spacetime_ndim == 2:
            return self._forward_2d(x)
        else:
            return self._forward_3d(x)

    def _forward_2d(self, x):
        """Forward pass for 2D inputs using 4-fold rotation averaging."""
        # Create 4 rotated copies
        rotations = [_rotate_2d(x, k) for k in range(4)]
        x_cat = torch.cat(rotations, dim=0)

        # Run through the base model
        out = self.base_model(x_cat)

        # Apply the column shift to break center-pixel access
        out = _shift_right(out)

        # Split back into 4 chunks and un-rotate
        chunks = torch.chunk(out, 4, dim=0)
        unrotated = [_rotate_2d(chunks[k], -k) for k in range(4)]

        # Average
        result = torch.stack(unrotated, dim=0).mean(dim=0)
        return result

    def _forward_3d(self, x):
        """Forward pass for 3D inputs using HW-plane rotation averaging."""
        # Use HW-plane rotations only (safe for any D/H/W combination)
        rotations = [_rotate_2d(x, k) for k in range(4)]
        x_cat = torch.cat(rotations, dim=0)

        out = self.base_model(x_cat)

        # Apply column shift
        out = _shift_right(out)

        # Split and un-rotate
        chunks = torch.chunk(out, 4, dim=0)
        unrotated = [_rotate_2d(chunks[k], -k) for k in range(4)]

        result = torch.stack(unrotated, dim=0).mean(dim=0)
        return result


def n2s_shiftconv_train(
    input_image,
    model: nn.Module,
    *,
    nb_epochs: int = 30,
    lr: float = 0.01,
    patience: int = 8,
    batch_size: int = 1,
    verbose: bool = True,
    stop_fitting_flag: dict = None,
):
    """Train a model using Noise2Self with the shift-convolution trick.

    Wraps the model with ``ShiftConvWrapper`` which rotates the input 4
    ways (2D), concatenates along the batch dimension, applies the model
    with a column shift, and averages un-rotated outputs.

    Parameters
    ----------
    input_image : numpy.ndarray
        Input noisy image tensor with shape ``(B, C, ...spatial_dims...)``.
    model : torch.nn.Module
        PyTorch model to train (e.g. UNet).
    nb_epochs : int
        Maximum number of training epochs.
    lr : float
        Initial learning rate for the AdamW optimizer.
    patience : int
        Number of epochs without improvement before early stopping.
    batch_size : int
        Batch size. Default is 1 (4 rotated copies use 4x memory).
    verbose : bool
        If ``True``, print loss values during training.
    stop_fitting_flag : dict, optional
        Mutable dict with a ``'stop'`` key for external stop.
    """
    device = get_torch_device()

    # Determine spatial ndim from input
    spacetime_ndim = input_image.ndim - 2

    # Wrap model with shift-conv
    wrapper = ShiftConvWrapper(model, spacetime_ndim=spacetime_ndim)
    wrapper = wrapper.to(device)

    optimizer = AdamW(wrapper.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=max(1, patience // 4))

    # Convert input to tensor
    x = torch.as_tensor(input_image, dtype=torch.float32).to(device)

    best_loss = math.inf
    best_model_state_dict = None
    patience_counter = 0

    wrapper.train()

    for epoch in range(nb_epochs):
        optimizer.zero_grad()

        output = wrapper(x)

        # MSE loss against the input (self-supervised)
        loss = F.mse_loss(output, x)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss = loss.item()
        scheduler.step(epoch_loss)

        if verbose:
            aprint(f"ShiftConv Loss ({epoch}): \t{epoch_loss:.8f}")

        # Best model tracking and early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_model_state_dict = OrderedDict(
                {k: v.to('cpu') for k, v in model.state_dict().items()}
            )
        else:
            patience_counter += 1
            if patience_counter > patience:
                aprint(f"Early stopping at epoch {epoch}.")
                break

        # Check external stop flag
        if stop_fitting_flag is not None and stop_fitting_flag.get('stop'):
            aprint("Training externally stopped.")
            break

    # Restore best model weights
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        aprint(f"Restored best model with loss {best_loss:.8f}")
