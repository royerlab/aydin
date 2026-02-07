"""Noise2Self self-supervised training method for PyTorch models.

Implements self-supervised training using grid-based pixel masking, where
the model learns to predict masked pixels from their unmasked neighbors.
"""

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
):
    """Train a model using the Noise2Self self-supervised method.

    Uses grid-based pixel masking where the model is trained to predict
    masked pixel values from their unmasked neighbors using MSE loss
    with AdamW optimizer and learning rate reduction on plateau.

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
        Number of epochs without improvement before stopping. Also
        controls the learning rate scheduler patience (patience // 8).
    verbose : bool
        If ``True``, print loss values during training.
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

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= max(num_batches, 1)
        scheduler.step(epoch_loss)

        if verbose:
            aprint("Loss (", epoch, "): \t", round(epoch_loss, 8))
