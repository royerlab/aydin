import random
from os.path import join
import keras.models
import numpy
from tensorflow.python.eager.context import device

from aydin.io.folders import get_temp_folder
from aydin.it.base import ImageTranslatorBase
from aydin.nn.models.jinet import JINetModel
from aydin.nn.models.unet import UNetModel
from aydin.nn.util.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    StopCenterGradient2D,
    StopCenterGradient3D,
)
from aydin.nn.util.data_util import random_sample_patches
from aydin.nn.util.validation_generator import train_image_generator
from aydin.regression.nn_utils.callbacks import ModelCheckpoint
from aydin.util.log.log import lsection, lprint
from aydin.util.tf.device import get_best_device_name, available_device_memory


class ImageTranslatorCNN(ImageTranslatorBase):
    """
    Convolutional Neural Network (CNN) based Image Translator<br>
    """

    verbose = 0

    def __init__(
        self,
        training_architecture: str = 'random',
        model_architecture: str = 'jinet',
        batch_size: int = 32,
        nb_unet_levels: int = 3,
        batch_norm: str = "instance",
        activation: str = 'ReLU',
        patch_size: int = 64,
        total_num_patches: int = None,
        adoption_rate: float = 0.5,
        mask_size: int = 5,
        random_mask_ratio: float = 0.1,
        max_epochs: int = 30,
        patience: int = 4,
        learn_rate: float = 0.01,
        **kwargs,
    ):
        """

        Parameters
        ----------
        training_architecture : str
            'shiftconv' or 'checkerbox' or 'random' or 'checkran' architecture
        model_architecture : str
            'unet' or 'jinet'
        batch_size : int
            Batch size for training
        nb_unet_levels : int
            Number of layers
        batch_norm
            Type of batch normalization (e.g. batch, instance)
        activation
        patch_size : int
            Size for patch sample e.g. 64 for (64, 64) or (64, 64, 64)
        total_num_patches
            Total number of patches for training
        adoption_rate
            % of random patches will be used for training, the rest will be discarded
        mask_size
            Mask shape for masking architecture; int of the same size as the spatial dimension
        random_mask_ratio
            Probability of masked pixels in random masking approach
        max_epochs : int
            Maximum number of epochs allowed
        patience : int
            Patience for EarlyStop or ReducedLR to be triggered
        learn_rate : float
            Initial learn rate
        kwargs
            Meant to have only keyword arguments for super class constructor. Do NOT abuse.
        """
        super().__init__(**kwargs)
        self.model_architecture = model_architecture  # both
        self.batch_size = batch_size  # both
        self.batch_norm = batch_norm  # both
        self.activation_fun = activation  # both
        self.patch_size = patch_size  # both
        self.total_num_patches = total_num_patches  # both
        self.adoption_rate = adoption_rate  # both
        self.max_epochs = max_epochs  # both
        self.patience = patience  # both
        self.learn_rate = learn_rate  # both
        self.model = None  # a CNN model  # both
        self.infmodel = None  # inference model  # both
        self.EStop_patience = self.patience * 2  # both
        self.ReduceLR_patience = self.patience  # both
        self.checkpoint = None  # both
        self.input_dim = None  # both
        self.stop_fitting = False  # both
        self.validation_images = None  # both
        self.validation_markers = None  # both
        self._create_patches_for_validation = (
            False  # if false use pixels for validation  # both
        )

        self.mask_size = mask_size  # unet
        self.random_mask_ratio = random_mask_ratio  # unet
        self.nb_unet_levels = nb_unet_levels  # unet
        self.training_architecture = training_architecture  # unet

        with lsection("CNN image translator"):
            lprint("training architecture: ", self.training_architecture)
            lprint("number of layers: ", self.nb_unet_levels)
            lprint("batch norm: ", self.batch_norm)
            lprint("mask size: ", self.mask_size)
            lprint("max_epochs", self.max_epochs)
            lprint("verbose: ", self.verbose)

    @property
    def model_class(self):
        if self.model_architecture == "jinet":
            return JINetModel
        elif self.model_architecture == "unet":
            return UNetModel
        else:
            raise ValueError("Unknown model architecture")

    def save(self, path: str):
        """
        Saves a 'all-batteries-included' image translation model at a given path (folder).

        Parameters
        ----------
        path : str
            path to save to

        Returns
        -------

        """
        with lsection(f"Saving 'CNN' image translator to {path}"):
            frozen = super().save(path)
            self.save_cnn(path)
        return frozen

    def save_cnn(self, path: str):
        if self.model is not None:
            # serialize model to JSON:
            self.model.save(join(path, "tf_model"))
            self.infmodel.save(join(path, "tf_inf_model"))
        else:
            lprint("There is no model to save yet.")

    def __getstate__(self):
        state = self.__dict__.copy()
        # exclude fields below that should/cannot be saved properly:
        del state['early_stopping']
        del state['reduce_learning_rate']
        del state['checkpoint']
        del state['model']
        del state['loss_history']
        del state['infmodel']
        del state['validation_images']

        return state

    def _load_internals(self, path: str):
        with lsection(f"Loading 'cnn' image translator from {path}"):
            # load JSON and create model:
            self.model = keras.models.load_model(join(path, "tf_model"))
            self.infmodel = keras.models.load_model(join(path, "tf_inf_model"))

    def get_receptive_field_radius(self, nb_unet_levels, shiftconv=False):
        """TODO: add proper docstrings here

        Parameters
        ----------
        nb_unet_levels : int
        shiftconv : bool

        Returns
        -------
        int

        """
        if shiftconv:
            rf = 7 if nb_unet_levels == 0 else 36 * 2 ** (nb_unet_levels - 1) - 6
        else:
            rf = 3 if nb_unet_levels == 0 else 18 * 2 ** (nb_unet_levels - 1) - 4
        return int(rf // 2)

    def stop_training(self):
        """Stops currently running training within the instance by turning the flag
        true for early stop callback.

        """
        self.stop_fitting = True

    def _train(
        self,
        input_image,
        target_image,
        train_valid_ratio=0.1,
        callback_period=3,
        jinv=False,
    ):
        with device(get_best_device_name()):
            # Reshape the input image
            input_image = numpy.moveaxis(input_image, 1, input_image.ndim - 1)
            if not self.self_supervised:
                target_image = numpy.moveaxis(target_image, 1, target_image.ndim - 1)

            self.spacetime_ndim = input_image.ndim - 2
            if self.spacetime_ndim not in [2, 3]:
                raise ValueError(
                    "Number of spacetime dimensions have to be either 2 or 3."
                )

            self.input_dim = input_image.shape[1:]

            # Compute patch size from batch size
            if self.patch_size is None:
                self.patch_size = (
                    self.get_receptive_field_radius(
                        self.nb_unet_levels,
                        shiftconv='shiftconv' == self.training_architecture,
                    )
                    * 2
                )

                self.patch_size = (
                    self.patch_size - self.patch_size % 2 ** self.nb_unet_levels
                )

                if self.patch_size < 2 ** self.nb_unet_levels:
                    raise ValueError(
                        'Number of layers is too large for given patch size.'
                    )

            lprint(f'Patch size: {self.patch_size}')
            # TODO: Do we need to have one if statement to automatically convert self.batch_size = 1 for shiftconv?
            if 'shiftconv' in self.training_architecture or (
                self.model_architecture == "jinet" and self.spacetime_ndim == 3
            ):
                self.batch_size = 1
                lprint(
                    'When patch_size is assigned under shiftconv architecture, batch_size is automatically set to 1.'
                )

            # Adjust patch_size for given input shape
            self.patch_size = [self.patch_size] * self.spacetime_ndim

            # Check patch_size for unet models
            if 'unet' in self.model_architecture:
                patch_size = numpy.array(self.patch_size)
                if (patch_size.max() / (2 ** self.nb_unet_levels) <= 0).any():
                    raise ValueError(
                        f'Tile size is too small. The largest dimension of tile size has to be >= {2 ** self.nb_unet_levels}.'
                    )
                if (patch_size[-2:] % 2 ** self.nb_unet_levels != 0).any():
                    raise ValueError(
                        f'Tile sizes on XY plane have to be multiple of 2^{self.nb_unet_levels}'
                    )

            # Check if the smallest dimension of input data >= patch_size
            if min(self.patch_size) > min(self.input_dim[:-1]):
                smallest_dim = min(self.input_dim[:-1])
                self.patch_size[numpy.argsort(self.input_dim[:-1])[0]] = (
                    smallest_dim // 2 * 2
                )

            # Determine total number of patches
            if self.total_num_patches is None:
                self.total_num_patches = min(
                    input_image.size / numpy.prod(self.patch_size), 10240
                )  # upper limit of num of patches
                self.total_num_patches = (
                    self.total_num_patches
                    - (self.total_num_patches % self.batch_size)
                    + self.batch_size
                )
            else:
                if self.total_num_patches < self.batch_size:
                    raise ValueError(
                        'total_num_patches has to be larger than batch_size.'
                    )
                self.total_num_patches = (
                    self.total_num_patches
                    - (self.total_num_patches % self.batch_size)
                    + self.batch_size
                )

            lprint(f"Available mem: {available_device_memory()}")
            lprint(f"Batch size for training: {self.batch_size}")

            # Decide whether to use validation pixels or patches
            if 1024 > input_image.size / numpy.prod(self.patch_size):
                with lsection(
                    f'Validation data will be created by monitoring {train_valid_ratio} of the pixels in the input data.'
                ):
                    img_train, img_val, val_marker = train_image_generator(
                        input_image, p=train_valid_ratio
                    )
            else:
                with lsection(
                    f'Validation data will be created by monitoring {train_valid_ratio} of the patches/images in the input data.'
                ):
                    self._create_patches_for_validation = True

            # Tile input and target image
            if self.patch_size is not None:
                with lsection('Random patch sampling...'):
                    lprint(f'Total number of patches: {self.total_num_patches}')
                    input_patch_idx = random_sample_patches(
                        input_image,
                        self.patch_size,
                        self.total_num_patches,
                        self.adoption_rate,
                    )

                    self.total_num_patches = len(input_patch_idx)

                    img_train_patch = []

                    if self._create_patches_for_validation:
                        for i in input_patch_idx:
                            img_train_patch.append(input_image[i])
                        img_train = numpy.vstack(img_train_patch)
                    else:
                        img_val_patch = []
                        marker_patch = []
                        for i in input_patch_idx:
                            img_train_patch.append(img_train[i])
                            img_val_patch.append(img_val[i])
                            marker_patch.append(val_marker[i])
                        img_train = numpy.vstack(img_train_patch)
                        img_val = numpy.vstack(img_val_patch)
                        val_marker = numpy.vstack(marker_patch)
                        self.validation_images = img_val
                        self.validation_markers = val_marker

                    if not self.self_supervised:
                        target_patch = []
                        for i in input_patch_idx:
                            target_patch.append(target_image[i])
                        target_image = numpy.vstack(target_patch)
                    else:
                        target_image = img_train

            # Last check of input size espetially for shiftconv
            if 'shiftconv' == self.training_architecture and self.self_supervised:
                # TODO: Hirofumi what is going on the conditional below <-- check input dim is compatible w/ shiftconv
                if (
                    numpy.mod(
                        img_train.shape[1:][:-1],
                        numpy.repeat(
                            2 ** self.nb_unet_levels, len(img_train.shape[1:][:-1])
                        ),
                    )
                    != 0
                ).any():
                    raise ValueError(
                        'Each dimension of the input image has to be a multiple of 2^nb_unet_levels for shiftconv.'
                    )
                lprint(
                    'Model will be generated for self-supervised learning with shift convolution scheme.'
                )
                if numpy.diff(img_train.shape[1:][:2]) != 0:
                    raise ValueError(
                        'Make sure the input image shape is cubic as shiftconv mode involves rotation.'
                    )
                if (
                    numpy.mod(
                        img_train.shape[1:][:-1],
                        numpy.repeat(
                            2 ** (self.nb_unet_levels - 1),
                            len(img_train.shape[1:][:-1]),
                        ),
                    )
                    != 0
                ).any():
                    raise ValueError(
                        'Each dimension of the input image has to be a multiple of '
                        '2^(nb_unet_levels-1) as shiftconv mode involvs pixel shift. '
                    )

            shiftconv = (
                'shiftconv' == self.training_architecture and self.self_supervised
            )

            self.model = self.model_class(
                img_train.shape[1:],
                spacetime_ndim=self.spacetime_ndim,
                mini_batch_size=self.batch_size,
                nb_unet_levels=self.nb_unet_levels,
                normalization=self.batch_norm,
                activation=self.activation_fun,
                supervised=not self.self_supervised,
                shiftconv=shiftconv,
                learning_rate=self.learn_rate,
            )

            with lsection('CNN model summary:'):
                lprint(f'Model architecture: {self.model_architecture}')
                lprint(f'Train scheme: {self.training_architecture}')
                if self.model_architecture == 'unet':
                    lprint(f'Number of layers: {self.nb_unet_levels}')
                lprint(
                    f'Number of parameters in the model: {self.model.count_params()}'
                )
                lprint(f'Batch normalization: {self.batch_norm}')
                lprint(f'Training input size: {img_train.shape[1:]}')

            # End of train function and beginning of _train from legacy implementation
            input_image = img_train

            with lsection(
                f"Training image translator from image of shape {input_image.shape} to image of shape {target_image.shape}:"
            ):

                if 'jinet' in self.model_architecture:
                    self.EStop_patience = self.EStop_patience + 10
                    self.ReduceLR_patience = self.ReduceLR_patience + 20

                # Early stopping patience:
                lprint(f"Early stopping patience: {self.EStop_patience}")

                # Effective LR patience:
                lprint(f"Effective LR patience: {self.ReduceLR_patience}")
                lprint(f'Batch size: {self.batch_size}')

                # Here is the list of callbacks:
                callbacks = []

                # Early stopping callback:
                self.early_stopping = EarlyStopping(
                    self, patience=self.EStop_patience, restore_best_weights=True
                )

                # Reduce LR on plateau:
                self.reduce_learning_rate = ReduceLROnPlateau(
                    verbose=1, patience=self.ReduceLR_patience, min_lr=1e-8, min_delta=0
                )
                self.reduce_learning_rate1 = ReduceLROnPlateau(
                    verbose=1,
                    patience=self.ReduceLR_patience,
                    min_lr=self.learn_rate * 0.01,
                    min_delta=0,
                )

                if self.checkpoint is None:
                    self.model_file_path = join(
                        get_temp_folder(),
                        f"aydin_cnn_keras_model_file_{random.randint(0, 1e16)}.hdf5",
                    )
                    lprint(f"Model will be saved at: {self.model_file_path}")
                    self.checkpoint = ModelCheckpoint(
                        self.model_file_path, verbose=1, save_best_only=True
                    )
                    # Add callbacks to the list:
                    callbacks.append(self.checkpoint)
                    callbacks.append(self.early_stopping)
                    callbacks.append(self.reduce_learning_rate)
                    if 'checkran' in self.training_architecture:
                        callbacks.append(self.reduce_learning_rate1)

                    if self.blind_spots:
                        self.blind_spots.remove((0, 0))

                    if self.spacetime_ndim == 2:
                        stop_center_gradient = StopCenterGradient2D(self.blind_spots)
                    elif self.spacetime_ndim == 3:
                        stop_center_gradient = StopCenterGradient3D(self.blind_spots)

                    callbacks = (
                        callbacks + [stop_center_gradient]
                        if 'jinet' in self.model_architecture and self.self_supervised
                        else callbacks
                    )

                # Convert mask_size to tuple
                self.mask_size = (self.mask_size,) * (input_image.ndim - 2)

                lprint("Training now...")
                if 'jinet' in self.model_architecture:
                    self.loss_history = self.model.fit(
                        input_image=input_image,
                        target_image=target_image,
                        max_epochs=self.max_epochs,
                        callbacks=callbacks,
                        verbose=self.verbose,
                        batch_size=self.batch_size,
                        total_num_patches=self.total_num_patches,
                        img_val=self.validation_images,
                        create_patches_for_validation=self._create_patches_for_validation,
                        train_valid_ratio=train_valid_ratio,
                    )
                else:
                    self.loss_history = self.model.fit(
                        input_image=input_image,
                        target_image=target_image,
                        max_epochs=self.max_epochs,
                        callbacks=callbacks,
                        verbose=self.verbose,
                        batch_size=self.batch_size,
                        total_num_patches=self.total_num_patches,
                        img_val=self.validation_images,
                        create_patches_for_validation=self._create_patches_for_validation,
                        train_valid_ratio=train_valid_ratio,
                        val_marker=self.validation_markers,
                        training_architecture=self.training_architecture,
                        random_mask_ratio=self.random_mask_ratio,
                        patch_size=self.patch_size,
                        mask_size=self.mask_size,
                        ReduceLR_patience=self.ReduceLR_patience,
                    )

    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        with device(get_best_device_name()):
            # Change dimensions to (B, space, C)
            input_image = numpy.moveaxis(input_image, 1, input_image.ndim - 1)

            # Check if padding is needed to have dim size of multiple of 2 in all dimension
            reshaped_for_cube = False
            reshaped_for_model = False
            spatial_shape = numpy.array(input_image.shape[1:-1])
            if abs(numpy.diff(spatial_shape)).min() != 0:
                reshaped_for_cube = True
                input_shape_max = numpy.ones(spatial_shape.shape) * spatial_shape.max()
                pad_square = (input_shape_max - spatial_shape) / 2
                pad_width1 = (
                    [[0, 0]]
                    + [
                        [
                            numpy.ceil(pad_square[i]).astype(int),
                            numpy.floor(pad_square[i]).astype(int),
                        ]
                        for i in range(len(pad_square))
                    ]
                    + [[0, 0]]
                )
                input_image = numpy.pad(input_image, pad_width1, 'edge')
                spatial_shape = numpy.array(input_image.shape[1:-1])

            if not (spatial_shape % 2 ** self.nb_unet_levels == 0).all():
                reshaped_for_model = True
                pad_width0 = (
                    2 ** self.nb_unet_levels
                    - (spatial_shape % 2 ** self.nb_unet_levels)
                    # + pad_square
                ) / 2
                pad_width2 = (
                    [[0, 0]]
                    + [
                        [
                            numpy.ceil(pad_width0[i]).astype(int),
                            numpy.floor(pad_width0[i]).astype(int),
                        ]
                        for i in range(len(pad_width0))
                    ]
                    + [[0, 0]]
                )
                input_image = numpy.pad(input_image, pad_width2, 'edge')

            # Change the batch_size in split layer or input dimensions accordingly
            kwargs_for_infmodel = {
                'spacetime_ndim': self.spacetime_ndim,
                'mini_batch_size': 1,
                'nb_unet_levels': self.nb_unet_levels,
                'normalization': self.batch_norm,
                'activation': self.activation_fun,
                'shiftconv': 'shiftconv' == self.training_architecture,
            }

            if len(input_image.shape[1:-1]) == 2:
                kwargs_for_infmodel['input_layer_size'] = [
                    None,
                    None,
                    input_image.shape[-1],
                ]
            elif len(input_image.shape[1:-1]) == 3:
                kwargs_for_infmodel['input_layer_size'] = [
                    None,
                    None,
                    None,
                    input_image.shape[-1],
                ]

                if self.model_architecture == "unet":
                    kwargs_for_infmodel['original_zdim'] = self.patch_size[0]

            if (
                'random' in self.training_architecture
                or 'check' in self.training_architecture
            ):
                kwargs_for_infmodel['supervised'] = True
            else:
                kwargs_for_infmodel['supervised'] = not self.self_supervised

            if self.infmodel is None:
                self.infmodel = self.model_class(**kwargs_for_infmodel)
            self.infmodel.set_weights(self.model.get_weights())

        try:
            output_image = self.infmodel.predict(
                input_image, batch_size=self.batch_size, verbose=self.verbose
            )
        except Exception:
            output_image = self.infmodel.predict(
                input_image, batch_size=1, verbose=self.verbose
            )

        # TODO: AhmetCan refactor
        if reshaped_for_model:
            if len(spatial_shape) == 2:
                output_image = output_image[
                    :,
                    pad_width2[1][0] : -pad_width2[1][1] or None,
                    pad_width2[2][0] : -pad_width2[2][1] or None,
                    :,
                ]
            else:
                output_image = output_image[
                    :,
                    pad_width2[1][0] : -pad_width2[1][1] or None,
                    pad_width2[2][0] : -pad_width2[2][1] or None,
                    pad_width2[3][0] : -pad_width2[3][1] or None,
                    :,
                ]
        if reshaped_for_cube:
            if len(spatial_shape) == 2:
                output_image = output_image[
                    :,
                    pad_width1[1][0] : -pad_width1[1][1] or None,
                    pad_width1[2][0] : -pad_width1[2][1] or None,
                    :,
                ]
            else:
                output_image = output_image[
                    :,
                    pad_width1[1][0] : -pad_width1[1][1] or None,
                    pad_width1[2][0] : -pad_width1[2][1] or None,
                    pad_width1[3][0] : -pad_width1[3][1] or None,
                    :,
                ]

        output_image = numpy.moveaxis(output_image, output_image.ndim - 1, 1)
        return output_image
