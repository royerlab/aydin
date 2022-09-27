import math
from collections import OrderedDict
from itertools import chain

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from aydin.nn.layers.custom_conv import double_conv_block
from aydin.nn.layers.pooling_down import PoolingDown
from aydin.nn.models.utils.masking.grid_masked_dataset import GridMaskedDataset
from aydin.nn.pytorch.optimizers.esadam import ESAdam
from aydin.util.log.log import lprint


class UNetModel(nn.Module):
    def __init__(
        self,
        spacetime_ndim,
        nb_unet_levels: int = 4,
        nb_filters: int = 8,
        learning_rate=0.01,
        pooling_mode: str = 'max',
    ):
        super(UNetModel, self).__init__()

        self.spacetime_ndim = spacetime_ndim
        self.nb_unet_levels = nb_unet_levels
        self.nb_filters = nb_filters
        self.learning_rate = learning_rate
        self.pooling_down = PoolingDown(spacetime_ndim, pooling_mode)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')

        self.double_conv_blocks_encoder = self._encoder_convolutions()

        self.unet_bottom_conv_out_channels = self.nb_filters * (
            2 ** (self.nb_unet_levels - 1)
        )
        self.unet_bottom_conv_block = double_conv_block(
            self.unet_bottom_conv_out_channels,
            self.unet_bottom_conv_out_channels * 2,
            self.unet_bottom_conv_out_channels,
            spacetime_ndim,
        )

        self.double_conv_blocks_decoder = self._decoder_convolutions()

        if spacetime_ndim == 2:
            self.final_conv = nn.Conv2d(self.nb_filters, 1, 1)
        else:
            self.final_conv = nn.Conv3d(self.nb_filters, 1, 1)

    def forward(self, x):
        """
        UNet forward method.

        Parameters
        ----------
        x
        input_mask : numpy.ArrayLike
            A mask per image must be passed with self-supervised training.

        Returns
        -------

        """
        skip_layer = []

        # Encoder
        for layer_index in range(self.nb_unet_levels):
            x = self.double_conv_blocks_encoder[layer_index](x)
            skip_layer.append(x)
            x = self.pooling_down(x)

        # Bottom
        x = self.unet_bottom_conv_block(x)

        # Decoder
        for layer_index in range(self.nb_unet_levels):
            x = self.upsampling(x)
            x = torch.cat([x, skip_layer.pop()], dim=1)
            x = self.double_conv_blocks_decoder[layer_index](x)

        # Final convolution
        x = self.final_conv(x)

        return x

    def _encoder_convolutions(self):
        convolution = nn.ModuleList()
        for layer_index in range(self.nb_unet_levels):
            if layer_index == 0:
                nb_filters_in = 1
                nb_filters_inner = self.nb_filters
                nb_filters_out = self.nb_filters
            else:
                nb_filters_in = self.nb_filters * (2 ** (layer_index - 1))
                nb_filters_inner = self.nb_filters * (2**layer_index)
                nb_filters_out = self.nb_filters * (2**layer_index)

            convolution.append(
                double_conv_block(
                    nb_filters_in,
                    nb_filters_inner,
                    nb_filters_out,
                    self.spacetime_ndim,
                )
            )

        return convolution

    def _decoder_convolutions(self):
        convolutions = nn.ModuleList()
        for layer_index in range(self.nb_unet_levels):
            if layer_index == self.nb_unet_levels - 1:
                nb_filters_in = self.nb_filters * 2
                nb_filters_inner = nb_filters_out = self.nb_filters
            else:
                nb_filters_in = self.nb_filters * (
                    2 ** (self.nb_unet_levels - layer_index)
                )
                nb_filters_inner = nb_filters_in // 2
                nb_filters_out = nb_filters_in // 4

            convolutions.append(
                double_conv_block(
                    nb_filters_in,
                    nb_filters_inner,
                    nb_filters_out,
                    spacetime_ndim=self.spacetime_ndim,
                    normalizations=(None, "batch"),
                )
            )

        return convolutions


def n2s_train(
    image,
    model: UNetModel,
    nb_epochs: int = 128,
    learning_rate: float = 0.001,
    # patch_size: int = 32,
    patience: int = 128
):
    """
    Noise2Self training method.

    Parameters
    ----------
    image
    model : UNetModel
    nb_epochs : int
    learning_rate : float
    patch_size : int
    patience : int

    """
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(dev)

    model = model.to(device)
    print(f"device {device}")

    optimizer = AdamW(model.parameters(), lr=learning_rate)

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

    dataset = GridMaskedDataset(image)
    print(f"dataset length: {len(dataset)}")
    data_loader = DataLoader(dataset, batch_size=16, num_workers=3, shuffle=False)

    model.train()

    for epoch in range(nb_epochs):
        loss = 0
        for i, batch in enumerate(data_loader):
            original_patch, net_input, mask = batch

            original_patch = original_patch.to(device)
            net_input = net_input.to(device)
            mask = mask.to(device)

            net_output = model(net_input)

            if epoch == 255:
                import napari

                viewer = napari.Viewer()
                viewer.add_image(
                    model(original_patch.to(device)).detach().cpu().numpy(),
                    name=f"{epoch}",
                )
                napari.run()

            loss = loss_function1(net_output * mask, original_patch * mask)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        scheduler.step(loss)

        print("Loss (", epoch, "): \t", round(loss.item(), 8))


def n2t_train(
    input_images,
    target_images,
    model: UNetModel,
    nb_epochs: int = 1024,
    learning_rate: float = 0.01,
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
    model : UNetModel
    nb_epochs : int
    learning_rate : float
    training_noise : float

    l2_weight_regularization
    patience
    patience_epsilon
    reduce_lr_factor
    reload_best_model_period
    best_val_loss_value

    """
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
