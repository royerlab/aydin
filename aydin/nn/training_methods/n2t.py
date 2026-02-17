"""Noise2Truth supervised training method for PyTorch models.

Implements supervised training with paired noisy/clean images using
L1 loss, early stopping, best-model checkpointing, and TensorBoard logging.
"""

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from aydin.nn.datasets.noisy_gt_dataset import NoisyGroundtruthDataset
from aydin.nn.optimizers.esadam import ESAdam
from aydin.util.log.log import aprint
from aydin.util.torch.device import get_torch_device


def n2t_train(
    input_images,
    target_images,
    model: nn.Module,
    *,
    nb_epochs: int = 1024,
    lr: float = 0.01,
    training_noise: float = 0.001,
    l2_weight_regularization=1e-9,
    patience=128,
    patience_epsilon=0.0,
    reduce_lr_factor=0.5,
    reload_best_model_period=1024,
    best_loss_value=None,
    use_tensorboard: bool = False,
    stop_fitting_flag: dict = None,
):
    """Train a model using the Noise2Truth supervised method.

    Uses paired noisy input and clean target images with L1 loss,
    early stopping based on training loss, and periodic reloading
    of the best model weights.

    Parameters
    ----------
    input_images : numpy.ndarray
        Noisy input image tensor with shape ``(B, C, ...spatial_dims...)``.
    target_images : numpy.ndarray
        Clean target image tensor with the same shape as ``input_images``.
    model : torch.nn.Module
        PyTorch model to train.
    nb_epochs : int
        Maximum number of training epochs.
    lr : float
        Initial learning rate for the ESAdam optimizer.
    training_noise : float
        Initial noise level added to gradients by the ESAdam optimizer.
    l2_weight_regularization : float
        L2 weight decay coefficient for the optimizer.
    patience : int
        Number of epochs without improvement before early stopping.
    patience_epsilon : float
        Minimum improvement required to reset the patience counter.
    reduce_lr_factor : float
        Factor by which the learning rate is reduced on plateau.
    reload_best_model_period : int
        Interval (in epochs) for reloading the best model weights
        when validation loss is not improving.
    best_loss_value : float or None
        Initial best loss value. If ``None``, starts at infinity.
    use_tensorboard : bool
        Whether to log training metrics to TensorBoard. Requires
        ``tensorboard`` to be installed.
    stop_fitting_flag : dict, optional
        Mutable dict with a ``'stop'`` key. If ``stop_fitting_flag['stop']``
        becomes ``True``, training is halted at the end of the current epoch.
    """
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter()
        except (ImportError, AttributeError):
            writer = None
    else:
        writer = None

    device = get_torch_device()

    model = model.to(device)

    reduce_lr_patience = patience // 2

    if best_loss_value is None:
        best_loss_value = math.inf

    optimizer = ESAdam(
        model.parameters(),
        lr=lr,
        start_noise_level=training_noise,
        weight_decay=l2_weight_regularization,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        'min',
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
    )

    def loss_function(u, v):
        """Compute the element-wise L1 (absolute difference) loss.

        Parameters
        ----------
        u : torch.Tensor
            Predicted tensor.
        v : torch.Tensor
            Target tensor.

        Returns
        -------
        torch.Tensor
            Element-wise absolute differences.
        """
        return torch.abs(u - v)

    dataset = NoisyGroundtruthDataset([input_images], [target_images], device=device)
    aprint(f"dataset length: {len(dataset)}")
    data_loader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=True)

    patience_counter = 0
    best_model_state_dict = None

    for epoch in range(nb_epochs):
        train_loss_value = 0
        num_batches = 0

        model.train()
        for i, batch in enumerate(data_loader):
            input_image, target_image = batch
            input_image = input_image.to(device)
            target_image = target_image.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass:
            translated_image = model(input_image)

            # translation loss (per voxel):
            translation_loss = loss_function(translated_image, target_image)

            # loss value (for all voxels):
            translation_loss_value = translation_loss.mean()

            # backpropagation:
            translation_loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Updating parameters
            optimizer.step()

            # Post-optimization hooks (for models like JINet)
            if hasattr(model, 'post_optimisation'):
                model.post_optimisation()

            # update training loss for whole image:
            train_loss_value += translation_loss_value.item()

            num_batches += 1

        train_loss_value /= num_batches
        aprint(f"Training loss value: {train_loss_value}")

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss_value, epoch)

        # Learning rate schedule:
        scheduler.step(train_loss_value)

        if train_loss_value < best_loss_value:
            aprint("## New best loss!")
            if train_loss_value < best_loss_value - patience_epsilon:
                aprint("## Good enough to reset patience!")
                patience_counter = 0

            # Update best loss value:
            best_loss_value = train_loss_value

            # Save model:
            best_model_state_dict = OrderedDict(
                {k: v.to('cpu') for k, v in model.state_dict().items()}
            )

        else:
            if epoch % max(1, reload_best_model_period) == 0 and best_model_state_dict:
                aprint("Reloading best model to date!")
                model.load_state_dict(best_model_state_dict)

            if patience_counter > patience:
                aprint("Early stopping!")
                break

            # No improvement:
            aprint(
                "No improvement of training loss, "
                f"patience = {patience_counter}/{patience} "
            )
            patience_counter += 1

        aprint(f"## Best loss: {best_loss_value}")

        # Check external stop flag
        if stop_fitting_flag is not None and stop_fitting_flag.get('stop'):
            aprint("Training externally stopped.")
            break

    # Restore best model weights
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        aprint(f"Restored best model with loss {best_loss_value}")

    if writer is not None:
        writer.flush()
        writer.close()
