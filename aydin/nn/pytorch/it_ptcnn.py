import math
from collections import OrderedDict
from itertools import chain

import numpy
import torch
from aydin.it.base import ImageTranslatorBase
from aydin.nn.pytorch.models.jidcnet import JInet2D
from aydin.nn.pytorch.models.masking import Masking
from aydin.nn.pytorch.optimizers.esadam import ESAdam
from aydin.util.array.nd import extract_tiles
from aydin.util.log.log import lsection, lprint
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset


def to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


class PTCNNImageTranslator(ImageTranslatorBase):
    """
    Pytorch-based CNN image translator
    """

    def __init__(
        self,
        max_epochs=1024,
        patience=128,
        patience_epsilon=0.0,
        learning_rate=0.01,
        batch_size=64,
        model_class=JInet2D,
        denoise_loss='l1',
        deconv_loss='l1',
        normaliser_type='percentile',
        balance_training_data=None,
        keep_ratio=1,
        max_voxels_for_training=4e6,
        use_cuda=True,
        device_index=0,
    ):
        """
        Constructs an image translator using the pytorch deep learning library.

        :param normaliser_type: normaliser type
        :param balance_training_data: balance data ? (limits number training entries per target value histogram bin)

        """
        super().__init__(normaliser_type)

        use_cuda = use_cuda and (torch.cuda.device_count() > 0)
        self.device = torch.device(f"cuda:{device_index}" if use_cuda else "cpu")
        lprint(f"Using device: {self.device}")

        self.max_epochs = max_epochs
        self.patience = patience
        self.patience_epsilon = patience_epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.denoise_loss = denoise_loss
        self.deconv_loss = deconv_loss
        self.max_voxels_for_training = max_voxels_for_training
        self.keep_ratio = keep_ratio
        self.balance_training_data = balance_training_data

        self.model_class = model_class

        # These parametres are 'private'  we don't expect the user to modify them,
        # but since they are fields they can be modified if need be after construction...
        self.l1_weight_regularisation = 1e-9
        self.l2_weight_regularisation = 1e-9
        self.training_noise = 0.001
        self.reload_best_model_period = max_epochs  # //2
        self.reduce_lr_patience = patience // 2  # TODO: consider patience//2
        self.reduce_lr_factor = 0.5
        self.masking = False
        self.enforce_blind_spot = True
        self.optimizer_class = ESAdam
        self.max_tile_size = 1024  # TODO: adjust based on available memory

        self._stop_training_flag = False

        # pytorch_info = torch.__config__.show().replace('\n', ';')
        # lprint(f"PyTorch Info: {pytorch_info}")

    def save(self, path: str):
        """
        Saves a 'all-batteries-included' image translation model at a given path (folder).
        :param path: path to save to
        """
        with lsection(f"Saving 'classic' image translator to {path}"):
            frozen = super().save(path)
            pass
            # TODO: complete!

        return frozen

    def _load_internals(self, path: str):
        with lsection(f"Loading 'classic' image translator from {path}"):
            pass
            # TODO: complete!

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['XXXXXXX']
        return state

    def stop_training(self):
        lprint("Received notification to stop training loop now.")
        self._stop_training_flag = True

    def _train(
        self,
        input_image,
        target_image,
        train_valid_ratio=0.1,
        callback_period=3,
        jinv=False,
    ):
        self._stop_training_flag = False

        shape = input_image.shape
        # num_batches = shape[0]
        num_input_channels = input_image.shape[1]
        num_output_channels = target_image.shape[1]
        # num_spatiotemp_dim = input_image.ndim - 2

        # tile size:
        tile_size = min(self.max_tile_size, min(shape[2:]))

        # Decide on how many voxels to be used for validation:
        num_val_voxels = int(train_valid_ratio * input_image.size)
        lprint(
            f"Number of voxels used for validation: {num_val_voxels} (train_valid_ratio={train_valid_ratio})"
        )

        # Generate random coordinates for these voxels:
        val_voxels = tuple(numpy.random.randint(d, size=num_val_voxels) for d in shape)
        lprint(f"Validation voxel coordinates: {val_voxels}")

        # Training Tile size:
        lprint(f"Train Tile dimensions: {tile_size}")

        # Prepare Training Dataset:
        dataset = self._get_dataset(
            input_image,
            target_image,
            self.self_supervised,
            tilesize=tile_size,
            mode='grid',
            validation_voxels=val_voxels,
        )
        lprint(f"Number tiles for training: {len(dataset)}")

        # Training Data Loader:
        # num_workers = max(3, os.cpu_count() // 2)
        num_workers = 0  # faster if data is already in memory...
        lprint(f"Number of workers for loading training/validation data: {num_workers}")
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Model
        self.model = self.model_class(num_input_channels, num_output_channels).to(
            self.device
        )

        number_of_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        lprint(
            f"Number of trainable parameters in {self.model_class} model: {number_of_parameters}"
        )

        if self.masking:
            self.masked_model = Masking(self.model).to(self.device)

        lprint(f"Optimizer class: {self.optimizer_class}")
        lprint(f"Learning rate : {self.learning_rate}")

        # Optimizer:
        optimizer = self.optimizer_class(
            chain(self.model.parameters()),
            lr=self.learning_rate,
            start_noise_level=self.training_noise,
            weight_decay=self.l2_weight_regularisation,
        )

        lprint(f"Optimizer: {optimizer}")

        # Denoise loss functon:
        loss_function = nn.L1Loss()
        if self.denoise_loss.lower() == 'l2':
            lprint("Training/Validation loss: L2")
            if self.masking:
                loss_function = (
                    lambda u, v, m: (u - v) ** 2 if m is None else ((u - v) * m) ** 2
                )
            else:
                loss_function = lambda u, v: (u - v) ** 2  # noqa : E731

        elif self.denoise_loss.lower() == 'l1':
            if self.masking:
                loss_function = (
                    lambda u, v, m: torch.abs(u - v)
                    if m is None
                    else torch.abs((u - v) * m)
                )
            else:
                loss_function = lambda u, v: torch.abs(u - v)  # noqa : E731
            lprint("Training/Validation loss: L1")

        # Start training:
        self._train_loop(data_loader, optimizer, loss_function)

    def _get_dataset(
        self,
        input_image: numpy.ndarray,
        target_image: numpy.ndarray,
        self_supervised: bool,
        tilesize: int,
        mode: str,
        validation_voxels,
    ):
        class _Dataset(Dataset):
            def __init__(self, input_image, target_image, tilesize):
                """ """

                num_channels_input = input_image.shape[1]
                num_channels_target = target_image.shape[1]

                def extract(image):
                    return extract_tiles(
                        image,
                        tile_size=tilesize,
                        extraction_step=tilesize,
                        flatten=True,
                    )

                bc_flat_input_image = input_image.reshape(-1, *input_image.shape[2:])
                bc_flat_input_tiles = numpy.concatenate(
                    [extract(x) for x in bc_flat_input_image]
                )
                self.input_tiles = bc_flat_input_tiles.reshape(
                    -1, num_channels_input, *bc_flat_input_tiles.shape[1:]
                )

                if self_supervised:
                    self.target_tiles = self.input_tiles
                else:
                    bc_flat_target_image = target_image.reshape(
                        -1, *target_image.shape[2:]
                    )
                    bc_flat_target_tiles = numpy.concatenate(
                        [extract(x) for x in bc_flat_target_image]
                    )
                    self.target_tiles = bc_flat_target_tiles.reshape(
                        -1, num_channels_target, *bc_flat_target_tiles.shape[1:]
                    )

                mask_image = numpy.zeros_like(input_image)
                mask_image[validation_voxels] = 1

                bc_flat_mask_image = mask_image.reshape(-1, *mask_image.shape[2:])
                bc_flat_mask_tiles = numpy.concatenate(
                    [extract(x) for x in bc_flat_mask_image]
                )
                self.mask_tiles = bc_flat_mask_tiles.reshape(
                    -1, num_channels_input, *bc_flat_mask_tiles.shape[1:]
                )

            def __len__(self):
                return len(self.input_tiles)

            def __getitem__(self, index):
                input = self.input_tiles[index, ...]
                target = self.target_tiles[index, ...]
                mask = self.mask_tiles[index, ...]

                return (input, target, mask)

        if mode == 'grid':
            return _Dataset(input_image, target_image, tilesize)
        else:
            return None

    def _train_loop(self, data_loader, optimizer, loss_function):

        # Scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            'min',
            factor=self.reduce_lr_factor,
            verbose=True,
            patience=self.reduce_lr_patience,
        )

        best_val_loss_value = math.inf
        best_model_state_dict = None
        patience_counter = 0

        with lsection("Training loop:"):
            lprint(f"Maximum number of epochs: {self.max_epochs}")
            lprint(
                f"Training type: {'self-supervised' if self.self_supervised else 'supervised'}"
            )

            for epoch in range(self.max_epochs):
                with lsection(f"Epoch {epoch}:"):

                    train_loss_value = 0
                    val_loss_value = 0
                    iteration = 0
                    for i, (input_images, target_images, val_mask_images) in enumerate(
                        data_loader
                    ):
                        lprint(f"index: {i}, shape:{input_images.shape}")

                        input_images_gpu = input_images.to(
                            self.device, non_blocking=True
                        )
                        target_images_gpu = target_images.to(
                            self.device, non_blocking=True
                        )
                        validation_mask_images_gpu = val_mask_images.to(
                            self.device, non_blocking=True
                        )

                        # Adding training noise to input:
                        if self.training_noise > 0:
                            with torch.no_grad():
                                alpha = self.training_noise / (
                                    1 + (10000 * epoch / self.max_epochs)
                                )
                                lprint(f"Training noise level: {alpha}")
                                training_noise = alpha * torch.randn_like(input_images)
                                input_images_gpu += training_noise.to(
                                    input_images_gpu.device
                                )

                        # Clear gradients w.r.t. parameters
                        optimizer.zero_grad()

                        # Forward pass:
                        self.model.train()
                        if self.masking:
                            translated_images_gpu = self.masked_model(input_images_gpu)
                        else:
                            translated_images_gpu = self.model(input_images_gpu)

                        # apply forward model:
                        forward_model_images_gpu = self._forward_model(
                            translated_images_gpu
                        )

                        # validation masking:
                        u = forward_model_images_gpu * (1 - validation_mask_images_gpu)
                        v = target_images_gpu * (1 - validation_mask_images_gpu)

                        # with napari.gui_qt():
                        #     viewer = napari.Viewer()
                        #     viewer.add_image(to_numpy(validation_mask_images_gpu), name='validation_mask_images_gpu')
                        #     viewer.add_image(to_numpy(forward_model_images_gpu), name='forward_model_images_gpu')
                        #     viewer.add_image(to_numpy(target_images_gpu), name='target_images_gpu')

                        # translation loss (per voxel):
                        if self.masking:
                            mask = self.masked_model.get_mask()
                            translation_loss = loss_function(u, v, mask)
                        else:
                            translation_loss = loss_function(u, v)

                        # loss value (for all voxels):
                        translation_loss_value = translation_loss.mean()

                        # Additional losses:
                        additional_loss_value = self._additional_losses(
                            translated_images_gpu, forward_model_images_gpu
                        )
                        if additional_loss_value is not None:
                            translation_loss_value += additional_loss_value

                        # backpropagation:
                        translation_loss_value.backward()

                        # Updating parameters
                        optimizer.step()

                        # post optimisation -- if needed:
                        self.model.post_optimisation()

                        # If self-supervised then enforce blind-spot:
                        if self.self_supervised and self.enforce_blind_spot:
                            try:
                                self.model.enforce_blind_spot()
                                lprint("Post optimisation corrections applied to model")
                            except AttributeError:
                                lprint(
                                    "NO post optimisation corrections applied to model"
                                )

                        # update training loss_deconvolution for whole image:
                        train_loss_value += translation_loss_value.item()
                        iteration += 1

                        # Validation:
                        with torch.no_grad():
                            # Forward pass:
                            self.model.eval()
                            if self.masking:
                                translated_images_gpu = self.masked_model(
                                    input_images_gpu
                                )
                            else:
                                translated_images_gpu = self.model(input_images_gpu)

                            # apply forward model:
                            forward_model_images_gpu = self._forward_model(
                                translated_images_gpu
                            )

                            # validation masking:
                            u = forward_model_images_gpu * validation_mask_images_gpu
                            v = target_images_gpu * validation_mask_images_gpu

                            # translation loss (per voxel):
                            if self.masking:
                                translation_loss = loss_function(u, v, None)
                            else:
                                translation_loss = loss_function(u, v)

                            # loss values:
                            translation_loss_value = (
                                translation_loss.mean().cpu().item()
                            )

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
                        if val_loss_value < best_val_loss_value - self.patience_epsilon:
                            lprint("## Good enough to reset patience!")
                            patience_counter = 0

                        # Update best val loss value:
                        best_val_loss_value = val_loss_value

                        # Save model:
                        best_model_state_dict = OrderedDict(
                            {k: v.to('cpu') for k, v in self.model.state_dict().items()}
                        )

                    else:
                        if (
                            epoch % max(1, self.reload_best_model_period) == 0
                            and best_model_state_dict
                        ):
                            lprint("Reloading best models to date!")
                            self.model.load_state_dict(best_model_state_dict)

                        if patience_counter > self.patience:
                            lprint("Early stopping!")
                            break

                        # No improvement:
                        lprint(
                            "No improvement of validation losses, patience = {patience_counter}/{self.patience} "
                        )
                        patience_counter += 1

                    lprint("## Best val loss: {best_val_loss_value}")

                    if self._stop_training_flag:
                        lprint("Training interupted!")
                        break

        lprint("Reloading best models to date!")
        self.model.load_state_dict(best_model_state_dict)

        if self.self_supervised and self.enforce_blind_spot:
            try:
                self.model.fill_blind_spot()
                lprint("Blind spot filled!")
            except AttributeError:
                lprint("Blind spot NOT filled! (no method available)")

    def _additional_losses(self, translated_image, forward_model_image):
        return None

    def _forward_model(self, input):
        return input

    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        """Internal method that translates an input image on the basis of the trained model.

        Parameters
        ----------
        input_image
            input image
        image_slice
        whole_image_shape

        Returns
        -------

        """
        input_image = torch.Tensor(input_image)
        input_image = input_image.to(self.device)
        inferred_image: torch.Tensor = self.model(input_image)
        inferred_image = inferred_image.detach().cpu().numpy()
        return inferred_image

    def visualise_weights(self):
        try:
            self.model.visualise_weights()
        except AttributeError:
            lprint(
                "Method 'visualise_weights()' unavailable, cannot visualise weights. "
            )
