import math
from collections import OrderedDict
from itertools import chain

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from aydin.nn.layers.conv_with_batch_norm import ConvWithBatchNorm
from aydin.nn.layers.pooling_down import PoolingDown
from aydin.nn.pytorch.optimizers.esadam import ESAdam
from aydin.util.log.log import lprint


class UNetModel(nn.Module):
    def __init__(
        self,
        spacetime_ndim,
        nb_unet_levels: int = 4,
        nb_filters: int = 8,
        learning_rate=0.01,
        supervised: bool = False,
        residual: bool = False,
        pooling_mode: str = 'max',
    ):
        super(UNetModel, self).__init__()

        self.nb_unet_levels = nb_unet_levels
        self.nb_filters = nb_filters
        self.learning_rate = learning_rate
        self.supervised = supervised
        self.residual = residual  # TODO: currently not implemented completely, not in use, worth to implement and test

        self.conv_with_batch_norms_first_conv_for_first_level = ConvWithBatchNorm(
            1, self.nb_filters, spacetime_ndim
        )

        self.conv_with_batch_norms_first_half = []
        for layer_index in range(self.nb_unet_levels):
            if (
                layer_index == 0
            ):  # Handle special case input dimensions for the first layer
                self.conv_with_batch_norms_first_half.append(
                    ConvWithBatchNorm(self.nb_filters, self.nb_filters, spacetime_ndim)
                )
            else:
                self.conv_with_batch_norms_first_half.append(
                    ConvWithBatchNorm(
                        self.nb_filters * layer_index,
                        self.nb_filters * (layer_index + 1),
                        spacetime_ndim,
                    )
                )

        # TODO: check if it is a bug to use this filter size on the bottom
        self.unet_bottom_conv_out_channels = self.nb_filters

        self.unet_bottom_conv_with_batch_norm = ConvWithBatchNorm(
            self.conv_with_batch_norms_first_half[-1].out_channels,
            self.unet_bottom_conv_out_channels,
            spacetime_ndim,
        )

        self.conv_with_batch_norms_second_half = []
        for layer_index in range(self.nb_unet_levels):
            if (
                layer_index == 0
            ):  # Handle special case input dimensions for the first layer
                self.conv_with_batch_norms_second_half.append(
                    ConvWithBatchNorm(
                        self.unet_bottom_conv_out_channels,
                        self.nb_filters
                        * max((self.nb_unet_levels - layer_index - 2), 1),
                        spacetime_ndim,
                    )
                )
            else:
                self.conv_with_batch_norms_second_half.append(
                    ConvWithBatchNorm(
                        self.nb_filters
                        * max((self.nb_unet_levels - layer_index - 1 - 2), 1),
                        self.nb_filters
                        * max((self.nb_unet_levels - layer_index - 2), 1),
                        spacetime_ndim,
                    )
                )

        self.pooling_down = PoolingDown(spacetime_ndim, pooling_mode)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')

        if spacetime_ndim == 2:
            self.conv = nn.Conv2d(8, 1, 1)
        else:
            self.conv = nn.Conv3d(8, 1, 1)

    def forward(self, x, input_msk=None):
        """
        UNet forward method.

        Parameters
        ----------
        x
        input_msk : numpy.ArrayLike
            A mask per image must be passed with self-supervised training.

        Returns
        -------

        """

        skip_layer = [x]

        x = self.conv_with_batch_norms_first_conv_for_first_level(x)

        for layer_index in range(self.nb_unet_levels):

            x = self.conv_with_batch_norms_first_half[layer_index](x)

            x = self.pooling_down(x)

            if layer_index != self.nb_unet_levels - 1:
                # print(f"skip layer added: x -> {x.shape}")
                skip_layer.append(x)

            # print("down")

        # print("before bottom")
        x = self.unet_bottom_conv_with_batch_norm(x)
        # print("after bottom")

        for layer_index in range(self.nb_unet_levels):
            x = self.upsampling(x)

            x = torch.add(x, skip_layer.pop())
            # if self.residual:
            #     x = torch.add(x, skip_layer.pop())
            # else:
            #     x = torch.cat([x, skip_layer.pop()], dim=1)

            x = self.conv_with_batch_norms_second_half[layer_index](x)

            x = self.conv_with_batch_norms_second_half[layer_index](x)

            # print("up")

        x = self.conv(x)

        if not self.supervised:
            if input_msk is not None:
                x *= input_msk
            else:
                raise ValueError(
                    "input_msk cannot be None for self-supervised training"
                )

        return x


def n2t_unet_train_loop(
    input_images,
    target_images,
    model: UNetModel,
    nb_epochs: int = 1024,
    learning_rate=0.01,
    training_noise=0.001,
    l2_weight_regularization=1e-9,
    patience=128,
    patience_epsilon=0.0,
    reduce_lr_factor=0.5,
    reload_best_model_period=1024,
    best_val_loss_value=None,
):
    writer = SummaryWriter()

    reduce_lr_patience = patience // 2

    if best_val_loss_value is None:
        best_val_loss_value = math.inf

    optimizer = ESAdam(
        chain(model.parameters()),
        lr=learning_rate,
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

    for epoch in range(nb_epochs):
        train_loss_value = 0
        val_loss_value = 0
        iteration = 0
        for i, (input_image, target_image) in enumerate(
            zip([input_images], [target_images])
        ):
            # print(f"index: {i}, shape:{input_image.shape}")

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
        lprint("Training loss value: {train_loss_value}")

        val_loss_value /= iteration
        lprint("Validation loss value: {val_loss_value}")

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
                "No improvement of validation losses, patience = {patience_counter}/{self.patience} "
            )
            patience_counter += 1

        lprint("## Best val loss: {best_val_loss_value}")

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
