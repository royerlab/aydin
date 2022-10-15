import math
from collections import OrderedDict
from itertools import chain
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from aydin.nn._legacy_pytorch.optimizers.esadam import ESAdam
from aydin.nn.datasets.noisy_gt_dataset import NoisyGroundtruthDataset

from aydin.util.log.log import lprint
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
    best_val_loss_value=None,
):
    """
    Noise2Truth training method.

    Parameters
    ----------
    input_images
    target_images
    model : nn.Module
    nb_epochs : int
    lr : float
    training_noise : float

    l2_weight_regularization
    patience
    patience_epsilon
    reduce_lr_factor
    reload_best_model_period
    best_val_loss_value

    """
    writer = SummaryWriter()

    device = get_torch_device()

    torch.autograd.set_detect_anomaly(True)

    model = model.to(device)

    reduce_lr_patience = patience // 2

    if best_val_loss_value is None:
        best_val_loss_value = math.inf

    optimizer = ESAdam(
        chain(model.parameters()),
        lr=lr,
        start_noise_level=training_noise,
        weight_decay=l2_weight_regularization,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        'min',
        factor=reduce_lr_factor,
        verbose=True,
        patience=reduce_lr_patience,
    )

    def loss_function(u, v):
        return torch.abs(u - v)

    dataset = NoisyGroundtruthDataset([input_images], [target_images], device=device)
    print(f"dataset length: {len(dataset)}")
    data_loader = DataLoader(dataset, batch_size=16, num_workers=3, shuffle=False)

    for epoch in range(nb_epochs):
        train_loss_value = 0
        val_loss_value = 0
        iteration = 0
        for i, batch in enumerate(data_loader):
            input_image, target_image = batch
            input_image = input_image.to(device)
            target_image = target_image.to(device)

            lprint(f"index: {i}, shape:{input_image.shape}")

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass:
            model.train()

            translated_image = model(input_image)

            # translation loss (per voxel):
            translation_loss = loss_function(translated_image, target_image)

            # loss value (for all voxels):
            translation_loss_value = translation_loss.mean()

            # backpropagation:
            translation_loss_value.backward()

            # Updating parameters
            optimizer.step()

            # update training loss_deconvolution for whole image:
            train_loss_value += translation_loss_value.item()
            iteration += 1

            # Validation:
            with torch.no_grad():
                # Forward pass:
                model.eval()

                translated_image = model(input_image)

                # translation loss (per voxel):
                translation_loss = loss_function(translated_image, target_image)

                # loss values:
                translation_loss_value = translation_loss.mean().cpu().item()

                # update validation loss_deconvolution for whole image:
                val_loss_value += translation_loss_value
                iteration += 1

        train_loss_value /= iteration
        lprint(f"Training loss value: {train_loss_value}")

        val_loss_value /= iteration
        lprint(f"Validation loss value: {val_loss_value}")

        writer.add_scalar("Loss/train", train_loss_value, epoch)
        writer.add_scalar("Loss/valid", val_loss_value, epoch)

        # Learning rate schedule:
        scheduler.step(val_loss_value)

        if val_loss_value < best_val_loss_value:
            lprint("## New best val loss!")
            if val_loss_value < best_val_loss_value - patience_epsilon:
                lprint("## Good enough to reset patience!")
                patience_counter = 0

            # Update best val loss value:
            best_val_loss_value = val_loss_value

            # Save model:
            best_model_state_dict = OrderedDict(
                {k: v.to('cpu') for k, v in model.state_dict().items()}
            )

        else:
            if epoch % max(1, reload_best_model_period) == 0 and best_model_state_dict:
                lprint("Reloading best models to date!")
                model.load_state_dict(best_model_state_dict)

            if patience_counter > patience:
                lprint("Early stopping!")
                break

            # No improvement:
            lprint(
                f"No improvement of validation losses, patience = {patience_counter}/{patience} "
            )
            patience_counter += 1

        lprint(f"## Best val loss: {best_val_loss_value}")

        # if epoch % 512 == 0:
        #     print(epoch)
        #     result = model(input_image)
        #
        #     with napari.gui_qt():
        #         viewer = napari.Viewer()
        #
        #         viewer.add_image(to_numpy(lizard_image), name='groundtruth')
        #         viewer.add_image(to_numpy(result), name=f'result-{epoch}')
        #         viewer.add_image(to_numpy(input_image), name='input')
        #
        #         viewer.grid.enabled = True

    writer.flush()
    writer.close()
