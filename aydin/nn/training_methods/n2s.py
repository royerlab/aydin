"""Noise2Self self-supervised training method for PyTorch models.

Implements self-supervised training using random pixel masking, where
the model learns to predict masked pixels from their unmasked neighbors.
"""

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from aydin.nn.datasets.random_masked_dataset import RandomMaskedDataset
from aydin.util.log.log import lprint
from aydin.util.torch.device import get_torch_device


def n2s_train(
    input_image,
    model: nn.Module,
    *,
    nb_epochs: int = 128,
    lr: float = 0.001,
    # patch_size: int = 32,
    patience: int = 128,
    verbose: bool = True,
):
    """Train a model using the Noise2Self self-supervised method.

    Uses random pixel masking where the model is trained to predict
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
    lprint(f"device {device}")

    optimizer = AdamW(model.parameters(), lr=lr)

    # optimizer = ESAdam(
    #     chain(model.parameters()),
    #     lr=learning_rate,
    #     start_noise_level=0.001,
    #     weight_decay=1e-9,
    # )

    scheduler = ReduceLROnPlateau(
        optimizer,
        'min',
        verbose=True,
        patience=patience // 8,
    )

    loss_function1 = MSELoss()

    dataset = RandomMaskedDataset(input_image)
    lprint(f"dataset length: {len(dataset)}")
    data_loader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=False)

    model.train()

    for epoch in range(nb_epochs):
        loss = 0
        for i, batch in enumerate(data_loader):
            original_patch, net_input, mask = batch

            original_patch = original_patch.to(device)
            net_input = net_input.to(device)
            mask = mask.to(device)

            net_output = model(net_input)

            loss = loss_function1(net_output * mask, original_patch * mask)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        scheduler.step(loss)

        if verbose:
            lprint("Loss (", epoch, "): \t", round(loss.item(), 8))
