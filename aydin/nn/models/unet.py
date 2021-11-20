import numpy
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import (
    Concatenate,
    Add,
    UpSampling3D,
    Cropping3D,
    ZeroPadding3D,
    UpSampling2D,
    ZeroPadding2D,
    Cropping2D,
    Conv2D,
    Conv3D,
)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.regularizers import l1

from aydin.nn.layers.util import Split, Rot90
from aydin.nn.layers.maskout import Maskout
from aydin.nn.models.utils.conv_block import (
    conv3d_bn,
    pooling_down3D,
    pooling_down2D,
    conv2d_bn,
)
from aydin.nn.models.utils.training_architectures import get_unet_fit_args
from aydin.nn.util.mask_generator import randmaskgen, maskedgen
from aydin.nn.util.validation_generator import val_data_generator
from aydin.util.log.log import lprint


class UNetModel(Model):
    """UNet model. Three training modes are available: supervised: noisy and clean images are required, shiftconv:
    self-supervised learning with shift and conv scheme non-shiftconv: self-supervised learning by masking pixels at
    each iteration
    """

    def __init__(
        self,
        input_layer_size,
        spacetime_ndim,
        training_architecture: str = 'random',
        mini_batch_size: int = 1,
        nb_unet_levels: int = 4,
        normalization: str = 'batch',  # None,  # 'instance',
        activation: str = 'ReLU',
        supervised: bool = False,
        shiftconv: bool = True,
        nfilters: int = 8,
        learning_rate: float = 0.01,
        original_zdim: int = None,
        weight_decay: int = 0,
        residual: bool = False,
        pooling_mode: str = 'max',
    ):
        """

        Parameters
        ----------
        input_layer_size
        spacetime_ndim
        mini_batch_size : int
            Mini-batch size
        nb_unet_levels : int
            Depth level of the UNet
        normalization : string
            normalization type, can be `batch` and `instance` for now
        activation : string
            Type of the activation function to use
        supervised : bool
            Flag that controls training approach
        shiftconv : bool
            Flag that controls use of shift convolutions
        nfilters : int
            Number of filters in first layer
        learning_rate : float
            Learning rate
        original_zdim : int
            Original Z-dimension length
        weight_decay : int
            coefficient of l1 regularizer
        residual : bool
            whether to use add or concat at merging layers
        pooling_mode : str
        """
        self.compiled = False

        self.training_architecture = training_architecture
        self.rot_batch_size = mini_batch_size
        self.num_lyr = nb_unet_levels
        self.normalization = normalization
        self.activation = activation
        self.shiftconv = shiftconv
        self.nfilters = nfilters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.residual = residual
        self.pooling_mode = pooling_mode

        if type(input_layer_size) is int:
            input_layer_size = (input_layer_size,) * spacetime_ndim + (1,)

        if spacetime_ndim == 3:
            if original_zdim:
                self.zdim = original_zdim
            else:
                self.zdim = input_layer_size[0]

        # Generate a model
        self.input_lyr = Input(input_layer_size, name='input')
        x = (
            self.unet_core_2d(self.input_lyr)
            if spacetime_ndim == 2
            else self.unet_core_3d()
        )

        if not shiftconv and not supervised:
            input_msk = Input(input_layer_size, name='input_msk')
            x = Maskout(name='maskout')([x, input_msk])
            super().__init__([self.input_lyr, input_msk], x)
        else:
            super().__init__(self.input_lyr, x)

        self.compile(optimizer=Adam(lr=learning_rate), loss='mse')
        self.compile = True

    def unet_core_2d(self, input_lyr):
        """Unet Core method which actually populates the model"""
        # Rotation & stack of the input images
        if self.shiftconv:
            input1 = Rot90(input_lyr, kk=1, lyrname='rot1')(input_lyr)
            input2 = Rot90(input_lyr, kk=2, lyrname='rot2')(input_lyr)
            input3 = Rot90(input_lyr, kk=3, lyrname='rot3')(input_lyr)
            x = Concatenate(name='conc_in', axis=0)([input_lyr, input1, input2, input3])
        else:
            x = input_lyr

        skiplyr = [x]
        for i in range(self.num_lyr):
            if i == 0:
                x = conv2d_bn(
                    x,
                    unit=self.nfilters * (i + 1),
                    shiftconv=self.shiftconv,
                    weight_decay=self.weight_decay,
                    lyrname=f'enc{i}_cv0',
                )

            x = conv2d_bn(
                x,
                unit=self.nfilters * (i + 1),
                shiftconv=self.shiftconv,
                norm=self.normalization,
                act=self.activation,
                weight_decay=self.weight_decay,
                lyrname=f'enc{i}',
            )
            x = pooling_down2D(
                x, self.shiftconv, mode=self.pooling_mode, lyrname=f'enc{i}pl'
            )
            if i != self.num_lyr - 1:
                skiplyr.append(x)

        x = conv2d_bn(
            x,
            self.nfilters,
            shiftconv=self.shiftconv,
            norm=self.normalization,
            act=self.activation,
            weight_decay=self.weight_decay,
            lyrname='bottm',
        )

        for i in range(self.num_lyr):
            x = UpSampling2D((2, 2), name=f'up{i}')(x)
            if self.residual:
                x = Add(name=f'add{i}')([x, skiplyr.pop()])
            else:
                x = Concatenate(name=f'cnct{i}')([x, skiplyr.pop()])
            x = conv2d_bn(
                x,
                self.nfilters * max((self.num_lyr - i - 2), 1),
                shiftconv=self.shiftconv,
                weight_decay=self.weight_decay,
                lyrname=f'dec{i}_cv0',
            )
            x = conv2d_bn(
                x,
                self.nfilters * max((self.num_lyr - i - 2), 1),
                shiftconv=self.shiftconv,
                norm=self.normalization,
                act=self.activation,
                weight_decay=self.weight_decay,
                lyrname=f'dec{i}',
            )

        if self.shiftconv:
            # Shift the center pixel
            x = ZeroPadding2D(((0, 0), (1, 0)), name='shiftc_pd')(x)
            x = Cropping2D(((0, 0), (0, 1)), name='shiftc_crp')(x)

            # Rotation & stack for the output
            output0 = Split(x, 0, self.rot_batch_size, 'split0')(x)
            output1 = Split(x, 1, self.rot_batch_size, 'split1')(x)
            output2 = Split(x, 2, self.rot_batch_size, 'split2')(x)
            output3 = Split(x, 3, self.rot_batch_size, 'split3')(x)
            output1 = Rot90(output1, -1, lyrname='rot4')(output1)
            output2 = Rot90(output2, -2, lyrname='rot5')(output2)
            output3 = Rot90(output3, -3, lyrname='rot6')(output3)
            x = Concatenate(name='cnct_last', axis=-1)(
                [output0, output1, output2, output3]
            )
            x = conv2d_bn(
                x,
                self.nfilters * 2 * 4,
                kernel_size=1,
                shiftconv=False,
                norm=self.normalization,
                act=self.activation,
                weight_decay=self.weight_decay,
                lyrname='last1',
            )
            x = conv2d_bn(
                x,
                self.nfilters,
                kernel_size=1,
                shiftconv=False,
                norm=self.normalization,
                act=self.activation,
                weight_decay=self.weight_decay,
                lyrname='last2',
            )

        x = Conv2D(
            1,
            (1, 1),
            padding='same',
            name='last0',
            kernel_regularizer=l1(self.weight_decay),
            bias_regularizer=l1(self.weight_decay),
            activation='linear',
        )(x)

        return x

    def unet_core_3d(self):
        """Unet Core method which actually populates the model"""
        # Rotation & stack of the input images
        if self.shiftconv:
            input1 = Rot90(self.input_lyr, kk=1, lyrname='rot1')(self.input_lyr)
            input2 = Rot90(self.input_lyr, kk=2, lyrname='rot2')(self.input_lyr)
            input3 = Rot90(self.input_lyr, kk=3, lyrname='rot3')(self.input_lyr)
            input5 = Rot90(self.input_lyr, kk=5, lyrname='rot5')(self.input_lyr)
            input6 = Rot90(self.input_lyr, kk=6, lyrname='rot6')(self.input_lyr)
            x = Concatenate(name='conc_in', axis=0)(
                [self.input_lyr, input1, input2, input3, input5, input6]
            )
        else:
            x = self.input_lyr

        skiplyr = [x]
        down2D_n = 0
        for i in range(self.num_lyr):
            if i == 0:
                x = conv3d_bn(
                    x,
                    unit=self.nfilters * (i + 1),
                    shiftconv=self.shiftconv,
                    weight_decay=self.weight_decay,
                    lyrname=f'enc{i}_cv0',
                )

            x = conv3d_bn(
                x,
                unit=self.nfilters * (i + 1),
                shiftconv=self.shiftconv,
                norm=self.normalization,
                act=self.activation,
                weight_decay=self.weight_decay,
                lyrname=f'enc{i}',
            )
            if self.zdim > 3:
                pool_size = (2, 2, 2)
            else:
                if self.shiftconv:
                    raise ValueError(
                        'Input size is too small against the depth of the CNN model. '
                        'Please use masking method or less num_lyr or larger input size.'
                    )
                else:
                    pool_size = (1, 2, 2)
                    down2D_n += 1

            if self.zdim % 2 != 0:
                x = tf.pad(
                    x,
                    tf.constant([[0, 0], [0, 1], [0, 0], [0, 0], [0, 0]]),
                    mode='REFLECT',
                )
                self.zdim = (self.zdim + 1) // 2
            else:
                self.zdim = self.zdim // 2

            x = pooling_down3D(x, self.shiftconv, pool_size, lyrname=f'enc{i}pl')

            if i != self.num_lyr - 1:
                skiplyr.append(x)

        x = conv3d_bn(
            x,
            self.nfilters,
            shiftconv=self.shiftconv,
            norm=self.normalization,
            act=self.activation,
            weight_decay=self.weight_decay,
            lyrname='bottm',  # * num_layer,
        )

        for i in range(self.num_lyr):
            if down2D_n > 0:
                x = UpSampling3D((1, 2, 2), name=f'up{i}')(x)
                down2D_n -= 1
            else:
                x = UpSampling3D((2, 2, 2), name=f'up{i}')(x)
            # Check if z dim of connecting layers from encoder & decoder are the same
            x_en = skiplyr.pop()
            if x_en.shape[1] != x.shape[1]:
                x = Cropping3D(((0, 1), (0, 0), (0, 0)), name=f'dec_crp{i}')(x)
            if self.residual:
                x = Add(name=f'add{i}')([x, x_en])
            else:
                x = Concatenate(name=f'cnct{i}')([x, x_en])
            x = conv3d_bn(
                x,
                self.nfilters * max((self.num_lyr - i - 2), 1),
                shiftconv=self.shiftconv,
                weight_decay=self.weight_decay,
                lyrname=f'dec{i}_cv0',
            )
            x = conv3d_bn(
                x,
                self.nfilters * max((self.num_lyr - i - 2), 1),
                shiftconv=self.shiftconv,
                norm=self.normalization,
                act=self.activation,
                weight_decay=self.weight_decay,
                lyrname=f'dec{i}',
            )

        if self.shiftconv:
            # Shift the center pixel
            x = ZeroPadding3D(((0, 0), (0, 0), (1, 0)), name='shiftc_pd')(x)
            x = Cropping3D(((0, 0), (0, 0), (0, 1)), name='shiftc_crp')(x)

            # Rotation & stack for the output
            output0 = Split(x, 0, self.rot_batch_size, 'split0')(x)
            output1 = Split(x, 1, self.rot_batch_size, 'split1')(x)
            output2 = Split(x, 2, self.rot_batch_size, 'split2')(x)
            output3 = Split(x, 3, self.rot_batch_size, 'split3')(x)
            output5 = Split(x, 4, self.rot_batch_size, 'split5')(x)
            output6 = Split(x, 5, self.rot_batch_size, 'split6')(x)
            output1 = Rot90(output1, -1, lyrname='rot-1')(output1)
            output2 = Rot90(output2, -2, lyrname='rot-2')(output2)
            output3 = Rot90(output3, -3, lyrname='rot-3')(output3)
            output5 = Rot90(output5, -5, lyrname='rot-5')(output5)
            output6 = Rot90(output6, -6, lyrname='rot-6')(output6)
            x = Concatenate(name='cnct_last', axis=-1)(
                [output0, output1, output2, output3, output5, output6]
            )
            x = conv3d_bn(
                x,
                self.nfilters * 2 * 4,
                kernel_size=3,  # a work around for a bug in tf; supposed to be 1
                shiftconv=False,
                norm=self.normalization,
                act=self.activation,
                weight_decay=self.weight_decay,
                lyrname='last1',
            )
            x = conv3d_bn(
                x,
                self.nfilters,
                kernel_size=3,  # a work around for a bug in tf; supposed to be 1
                shiftconv=False,
                norm=self.normalization,
                act=self.activation,
                weight_decay=self.weight_decay,
                lyrname='last2',
            )

        x = Conv3D(
            1,
            3,  # a work around for a bug in tf; supposed to be 1
            padding='same',
            name='last0',
            kernel_regularizer=l1(self.weight_decay),
            bias_regularizer=l1(self.weight_decay),
            activation='linear',
        )(x)

        return x

    def size(self):
        """Returns size of the model in bytes"""
        return self.count_params() * 4

    def fit(
        self,
        input_image=None,
        target_image=None,
        callbacks=None,
        train_valid_ratio=None,
        img_val=None,
        val_marker=None,
        training_architecture=None,
        create_patches_for_validation=None,
        total_num_patches=None,
        batch_size=None,
        random_mask_ratio=None,
        patch_size=None,
        mask_size=None,
        replace_by="zero",
        verbose=None,
        max_epochs=None,
        ReduceLR_patience=None,
        reduce_lr_factor=0.3,
    ):
        """

        Parameters
        ----------
        input_image
        target_image
        callbacks
        train_valid_ratio
        img_val
        val_marker
        training_architecture
        create_patches_for_validation
        total_num_patches
        batch_size
        random_mask_ratio
        patch_size
        mask_size
        replace_by
        verbose
        max_epochs
        ReduceLR_patience
        reduce_lr_factor

        Returns
        -------
        loss_history

        """
        if create_patches_for_validation:
            tv_ratio = train_valid_ratio
        else:
            tv_ratio = 0

        validation_data, validation_steps = get_unet_fit_args(
            train_method=training_architecture,
            create_patches_for_validation=create_patches_for_validation,
            input_image=input_image,
            total_num_patches=total_num_patches,
            train_valid_ratio=train_valid_ratio,
            batch_size=batch_size,
            random_mask_ratio=random_mask_ratio,
            img_val=img_val,
            patch_size=patch_size,
            mask_size=mask_size,
            val_marker=val_marker,
            replace_by=replace_by,
        )

        if 'shiftconv' in training_architecture:
            loss_history = super().fit(
                input_image,
                target_image,
                epochs=max_epochs,
                callbacks=callbacks,
                verbose=verbose,
                batch_size=batch_size,
                validation_data=validation_data,
            )
        elif 'checkerbox' in training_architecture:
            loss_history = super().fit(
                maskedgen(input_image, batch_size, mask_size, replace_by=replace_by),
                epochs=max_epochs,
                steps_per_epoch=numpy.prod(mask_size)
                * numpy.ceil(input_image.shape[0] * (1 - tv_ratio) / batch_size).astype(
                    int
                ),
                verbose=verbose,
                callbacks=callbacks,
                validation_data=validation_data,
                validation_steps=validation_steps,
            )
        elif 'random' in training_architecture:
            loss_history = super().fit(
                randmaskgen(
                    input_image,
                    batch_size,
                    p_maskedpixels=random_mask_ratio,
                    replace_by=replace_by,
                ),
                epochs=max_epochs,
                steps_per_epoch=numpy.ceil(1 / random_mask_ratio).astype(int)
                * numpy.ceil(total_num_patches * (1 - tv_ratio) / batch_size).astype(
                    int
                ),
                verbose=verbose,
                callbacks=callbacks,
                validation_data=validation_data,
                validation_steps=validation_steps,
            )
        elif 'checkran' in training_architecture:
            # train with checkerbox first
            lprint('Starting with checkerbox masking.')
            history_checkerbox = super().fit(
                maskedgen(input_image, batch_size, mask_size, replace_by=replace_by),
                epochs=max_epochs,
                steps_per_epoch=numpy.prod(mask_size)
                * numpy.ceil(input_image.shape[0] * (1 - tv_ratio) / batch_size).astype(
                    int
                ),
                verbose=verbose,
                callbacks=callbacks[:-2] + callbacks[-1:],
                validation_data=maskedgen(
                    input_image, batch_size, mask_size, replace_by=replace_by
                )
                if create_patches_for_validation
                else val_data_generator(
                    input_image,
                    img_val,
                    val_marker,
                    batch_size,
                    train_valid_ratio=train_valid_ratio,
                ),
                validation_steps=max(
                    numpy.ceil(
                        input_image.shape[0] * train_valid_ratio / batch_size
                    ).astype(int),
                    1,
                ),
            )

            # Then switch to random masking
            lprint('Switched to random masking.')
            history_random = super().fit(
                randmaskgen(
                    input_image,
                    batch_size,
                    p_maskedpixels=random_mask_ratio,
                    replace_by=replace_by,
                ),
                epochs=max_epochs,
                steps_per_epoch=numpy.ceil(1 / random_mask_ratio).astype(int)
                * numpy.ceil(total_num_patches * (1 - tv_ratio) / batch_size).astype(
                    int
                ),
                verbose=verbose,
                callbacks=callbacks[:-1],
                initial_epoch=history_checkerbox.epoch[-1] + 1,
                validation_data=randmaskgen(
                    input_image,
                    batch_size,
                    p_maskedpixels=random_mask_ratio,
                    replace_by=replace_by,
                )
                if create_patches_for_validation
                else val_data_generator(
                    input_image,
                    img_val,
                    val_marker,
                    batch_size,
                    train_valid_ratio=train_valid_ratio,
                ),
                validation_steps=max(
                    numpy.ceil(
                        total_num_patches * train_valid_ratio / batch_size
                    ).astype(int),
                    1,
                ),
            )
            history_checkerbox.epoch += history_random.epoch
            for key, val in history_random.history.items():
                history_checkerbox.history[key] += history_random.history[key]
            loss_history = history_checkerbox
        else:
            loss_history = -1

        return loss_history

    def predict(
        self,
        x,
        batch_size=None,
        verbose=0,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
        """Overwritten model predict method.

        Parameters
        ----------
        x
        batch_size
        verbose
        steps
        callbacks
        max_queue_size
        workers
        use_multiprocessing

        Returns
        -------

        """
        # TODO: move as much as you can from it cnn _translate
        return super().predict(
            x,
            batch_size=batch_size,
            verbose=verbose,
        )
