from itertools import chain

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from aydin.io.datasets import lizard, add_noise
from aydin.nn.models.torch_unet import UNetModel
from aydin.nn.models.utils.torch_dataset import TorchDataset
from aydin.nn.pytorch.optimizers.esadam import ESAdam
from aydin.util.log.log import lprint


def test_supervised_2D():
    input_array = torch.zeros((1, 1, 64, 64))
    model2d = UNetModel(
        # (64, 64, 1),
        nb_unet_levels=2,
        supervised=True,
        spacetime_ndim=2,
        residual=True,
    )
    result = model2d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_supervised_2D_n2t():
    lizard_image = lizard()
    input_image = add_noise(lizard_image)

    model = UNetModel(
        # (64, 64, 1),
        nb_unet_levels=2,
        supervised=True,
        spacetime_ndim=2,
        residual=True,
    )

    dataset = TorchDataset(
        input_image,
        lizard_image,
        64,
        self_supervised=False,
        validation_voxels=val_voxels,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = ESAdam(
        chain(model.parameters()),
        lr=learning_rate,
        start_noise_level=training_noise,
        weight_decay=l2_weight_regularisation,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        'min',
        factor=reduce_lr_factor,
        verbose=True,
        patience=reduce_lr_patience,
    )

    for epoch in range(1024):
        train_loss_value = 0
        val_loss_value = 0
        iteration = 0
        for i, (input_images, target_images, validation_mask_images) in enumerate(
            data_loader
        ):
            lprint(f"index: {i}, shape:{input_images.shape}")

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass:
            model.train()

            translated_images = model(input_images)

            # apply forward model:
            forward_model_images = _forward_model(translated_images)

            # validation masking:
            u = forward_model_images * (1 - validation_mask_images)
            v = target_images * (1 - validation_mask_images)

            # translation loss (per voxel):
            translation_loss = loss_function(u, v)

            # loss value (for all voxels):
            translation_loss_value = translation_loss.mean()

            # Additional losses:
            additional_loss_value = _additional_losses(
                translated_images, forward_model_images
            )
            if additional_loss_value is not None:
                translation_loss_value += additional_loss_value

            # backpropagation:
            translation_loss_value.backward()

            # Updating parameters
            optimizer.step()

            # post optimisation -- if needed:
            model.post_optimisation()

            # update training loss_deconvolution for whole image:
            train_loss_value += translation_loss_value.item()
            iteration += 1

            # Validation:
            with torch.no_grad():
                # Forward pass:
                model.eval()

                translated_images = model(input_images)

                # apply forward model:
                forward_model_images = _forward_model(translated_images)

                # validation masking:
                u = forward_model_images * validation_mask_images
                v = target_images * validation_mask_images

                # translation loss (per voxel):
                translation_loss = loss_function(u, v)

                # loss values:
                translation_loss_value = translation_loss.mean().cpu().item()

                # update validation loss_deconvolution for whole image:
                val_loss_value += translation_loss_value
                iteration += 1

        train_loss_value /= iteration
        lprint("Training loss value: {train_loss_value}")

        val_loss_value /= iteration
        lprint("Validation loss value: {val_loss_value}")

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
                "No improvement of validation losses, patience = {patience_counter}/{self.patience} "
            )
            patience_counter += 1

        lprint("## Best val loss: {best_val_loss_value}")

    result = model(input_image)
    assert result.shape == input_image.shape
    assert result.dtype == input_image.dtype


def test_masking_2D():
    input_array = torch.zeros((1, 1, 64, 64))
    model2d = UNetModel(
        # (64, 64, 1),
        nb_unet_levels=2,
        supervised=False,
        spacetime_ndim=2,
    )
    result = model2d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


# def test_jinet_2D():
#     input_array = torch.zeros((1, 1, 64, 64))
#     model2d = JINetModel((64, 64, 1), spacetime_ndim=2)
#     result = model2d.predict([input_array])
#     assert result.shape == input_array.shape
#     assert result.dtype == input_array.dtype


def test_supervised_3D():
    input_array = torch.zeros((1, 1, 64, 64, 64))
    model3d = UNetModel(
        # (64, 64, 64, 1),
        nb_unet_levels=2,
        supervised=True,
        spacetime_ndim=3,
    )
    result = model3d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_masking_3D():
    input_array = torch.zeros((1, 1, 64, 64, 64))
    model3d = UNetModel(
        # (64, 64, 64, 1),
        nb_unet_levels=2,
        supervised=False,
        spacetime_ndim=3,
    )
    result = model3d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


# def test_various_masking_3D():
#     for i in [0, 4]:
#         input_array = torch.zeros((1, 21 + i, 64, 64, 1))
#         print(f'input shape: {input_array.shape}')
#         model3d = UNetModel(
#             input_array.shape[1:],
#             nb_unet_levels=4,
#             supervised=False,
#             spacetime_ndim=3,
#         )
#         result = model3d.predict([input_array, input_array])
#         assert result.shape == input_array.shape
#         assert result.dtype == input_array.dtype
#
#
# def test_thin_masking_3D():
#     for i in range(3):
#         input_array = torch.zeros((1, 2 + i, 64, 64, 1))
#         print(f'input shape: {input_array.shape}')
#         model3d = UNetModel(
#             input_array.shape[1:],
#             nb_unet_levels=4,
#             supervised=False,
#             spacetime_ndim=3,
#         )
#         result = model3d.predict([input_array, input_array])
#         assert result.shape == input_array.shape
#         assert result.dtype == input_array.dtype
