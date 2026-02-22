"""Noise2Self self-supervised training method for PyTorch models.

Implements self-supervised training using grid-based pixel masking, where
the model learns to predict masked pixels from their unmasked neighbors.
"""

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from aydin.nn.datasets.grid_masked_dataset import GridMaskedDataset
from aydin.util.log.log import aprint
from aydin.util.torch.device import get_torch_device


def n2s_train(
    input_image,
    model: nn.Module,
    *,
    nb_epochs: int = 128,
    lr: float = 0.001,
    patience: int = 128,
    verbose: bool = True,
    stop_fitting_flag: dict = None,
):
    """Train a model using the Noise2Self self-supervised method.

    Uses grid-based pixel masking where the model is trained to predict
    masked pixel values from their unmasked neighbors using MSE loss
    with AdamW optimizer, learning rate reduction on plateau, early
    stopping with best-model tracking, and optional external stop flag.

    Parameters
    ----------
    input_image : numpy.ndarray
        Input noisy image tensor with shape ``(B, C, ...spatial_dims...)``.
    model : torch.nn.Module
        PyTorch model to train.
    nb_epochs : int
        Maximum number of training epochs.
    lr : float
        Initial learning rate for the AdamW optimizer.
    patience : int
        Number of epochs without improvement before early stopping.
        Also controls the learning rate scheduler patience (patience // 8).
    verbose : bool
        If ``True``, print loss values during training.
    stop_fitting_flag : dict, optional
        Mutable dict with a ``'stop'`` key. If ``stop_fitting_flag['stop']``
        becomes ``True``, training is halted at the end of the current epoch.
    """
    device = get_torch_device()

    model = model.to(device)
    aprint(f"device {device}")

    optimizer = AdamW(model.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(
        optimizer,
        'min',
        patience=patience // 8,
    )

    dataset = GridMaskedDataset(input_image)
    aprint(f"dataset length: {len(dataset)}")
    data_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    best_loss = math.inf
    best_model_state_dict = None
    patience_counter = 0

    model.train()

    for epoch in range(nb_epochs):
        epoch_loss = 0
        num_batches = 0
        for i, batch in enumerate(data_loader):
            original_patch, net_input, mask = batch

            original_patch = original_patch.to(device)
            net_input = net_input.to(device)
            mask = mask.to(device)

            net_output = model(net_input)

            # Compute MSE only over blind-spot pixels to avoid loss dilution
            diff = (net_output - original_patch) * mask
            mask_count = mask.sum()
            loss = diff.pow(2).sum() / mask_count.clamp(min=1)

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Post-optimization hooks (for models like JINet)
            if hasattr(model, 'post_optimisation'):
                model.post_optimisation()
            if hasattr(model, 'enforce_blind_spot'):
                model.enforce_blind_spot()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= max(num_batches, 1)
        scheduler.step(epoch_loss)

        if verbose:
            aprint("Loss (", epoch, "): \t", round(epoch_loss, 8))

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

            if verbose:
                aprint(f"No improvement, patience = {patience_counter}/{patience}")

        # Check external stop flag
        if stop_fitting_flag is not None and stop_fitting_flag.get('stop'):
            aprint("Training externally stopped.")
            break

    # Restore best model weights
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        aprint(f"Restored best model with loss {best_loss:.8f}")

    # Fill blind spot after training (for JINet-style models)
    if hasattr(model, 'fill_blind_spot'):
        model.fill_blind_spot()
