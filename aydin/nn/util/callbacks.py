import warnings
import numpy as np
from tensorflow.python.keras import backend
from tensorflow.python.keras.callbacks import Callback

from aydin.util.log.log import lprint


class CNNCallback(Callback):
    """
    tensorflow.keras Callback to linkup to the it callback machinery

    Parameters
    ----------
    monitor_callback
    """

    def __init__(self, monitor_callback):
        super().__init__()

        self.monitor_callback = monitor_callback
        self.iteration = 0

    def on_train_begin(self, logs=None):
        pass

    def on_batch_ends(self, batch, logs=None):
        self.notify(logs)

    def on_epoch_end(self, epoch, logs=None):
        self.notify(logs)

    def on_train_end(self, logs=None):
        self.notify(logs)

    def notify(self, logs):
        if self.monitor_callback:
            iteration = self.iteration
            val_loss = self.get_monitor_value(logs)
            self.monitor_callback(iteration, val_loss)
            self.iteration += 1
        pass

    def get_monitor_value(self, logs):
        if logs:
            if logs.get('val_loss'):
                monitor_value = logs.get('val_loss')
            else:
                monitor_value = logs.get('loss')
        else:
            monitor_value = -1
            # monitor_value = self.params['metrics']
        return monitor_value


class EarlyStopping(Callback):
    """
    Stop training when a monitored quantity has stopped improving.

    Parameters
    ----------
    parent
    monitor
        quantity to be monitored
    min_delta
        minimum change in the monitored quantity
        to qualify as an improvement, i.e. an absolute
        change of less than min_delta, will count as no
        improvement
    patience
        number of epochs with no improvement
        after which training will be stopped
    mode
        one of {auto, min, max}. In `min` mode,
        training will stop when the quantity
        monitored has stopped decreasing; in `max`
        mode it will stop when the quantity
        monitored has stopped increasing; in `auto`
        mode, the direction is automatically inferred
        from the name of the monitored quantity
    baseline
        Baseline value for the monitored quantity to reach.
        Training will stop if the model doesn't show improvement
        over the baseline
    restore_best_weights
        whether to restore model weights from
        the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of
        training are used.
    """

    def __init__(
        self,
        parent,
        monitor='val_loss',
        min_delta=0,
        patience=0,
        mode='auto',
        baseline=None,
        restore_best_weights=False,
    ):
        super(EarlyStopping, self).__init__()

        self.parent = parent
        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn(
                'EarlyStopping mode %s is unknown, ' 'fallback to auto mode.' % mode,
                RuntimeWarning,
            )
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        """on_train_begin part of callback"""
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        """on_epoch_end part of callback"""
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    lprint('Restoring model weights from the end of ' 'the best epoch')
                    self.model.set_weights(self.best_weights)

        # This is where we stop training externally:
        if self.parent.stop_fitting:
            lprint('Training externally stopped!')
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        """on_train_end part of callback"""
        if self.stopped_epoch > 0:
            lprint('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        """returns the monitor value"""
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s'
                % (self.monitor, ','.join(list(logs.keys()))),
                RuntimeWarning,
            )
        return monitor_value


class ReduceLROnPlateau(Callback):
    """
    Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Example
    -------

    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```

    Parameters
    ----------
    monitor
        quantity to be monitored
    factor
        factor by which the learning rate will
        be reduced. new_lr = lr * factor
    patience
        number of epochs with no improvement
        after which learning rate will be reduced
    verbose : int
        0: quiet, 1: update messages
    mode
        one of {auto, min, max}. In `min` mode,
        lr will be reduced when the quantity
        monitored has stopped decreasing; in `max`
        mode it will be reduced when the quantity
        monitored has stopped increasing; in `auto`
        mode, the direction is automatically inferred
        from the name of the monitored quantity
    min_delta
        threshold for measuring the new optimum,
        to only focus on significant changes
    cooldown
        number of epochs to wait before resuming
        normal operation after lr has been reduced
    min_lr
        lower bound on the learning rate
    kwargs
    """

    def __init__(
        self,
        monitor='val_loss',
        factor=0.1,
        patience=10,
        verbose=0,
        mode='auto',
        min_delta=1e-4,
        cooldown=0,
        min_lr=0,
        **kwargs,
    ):
        super(ReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            warnings.warn(
                '`epsilon` argument is deprecated and '
                'will be removed, use `min_delta` instead.'
            )
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn(
                'Learning Rate Plateau Reducing mode %s is unknown, '
                'fallback to auto mode.' % (self.mode),
                RuntimeWarning,
            )
            self.mode = 'auto'
        if self.mode == 'min' or (self.mode == 'auto' and 'acc' not in self.monitor):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s'
                % (self.monitor, ','.join(list(logs.keys()))),
                RuntimeWarning,
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(backend.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        backend.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            lprint(
                                'Epoch %05d: ReduceLROnPlateau reducing '
                                'learning rate to %s.' % (epoch + 1, new_lr)
                            )
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


class ModelCheckpoint(Callback):
    """
    Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    Parameters
    ----------
    filepath : string
        path to save the model file.
    monitor
        quantity to monitor
    verbose
        verbosity mode, 0 or 1.
    save_best_only
        if `save_best_only=True`,
        the latest best model according to
        the quantity monitored will not be overwritten.
    save_weights_only
        if True, then only the model's weights will be
        saved (`model.save_weights(filepath)`), else the full model
        is saved (`model.save(filepath)`).
    mode
        one of {auto, min, max}.
        If `save_best_only=True`, the decision
        to overwrite the current save file is made
        based on either the maximization or the
        minimization of the monitored quantity. For `val_acc`,
        this should be `max`, for `val_loss` this should
        be `min`, etc. In `auto` mode, the direction is
        automatically inferred from the name of the monitored quantity.
    period
        Interval (number of epochs) between checkpoints.
    """

    def __init__(
        self,
        filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1,
    ):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn(
                'ModelCheckpoint mode %s is unknown, '
                'fallback to auto mode.' % (mode),
                RuntimeWarning,
            )
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        'Can save best model only with %s available, '
                        'skipping.' % (self.monitor),
                        RuntimeWarning,
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            lprint(
                                'Epoch %05d: %s=%0.5f improved from %0.5f to %0.5f,'
                                % (epoch + 1, self.monitor, current, self.best, current)
                            )
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            lprint(
                                'Epoch %05d: %s=%0.5f did not improve from %0.5f'
                                % (epoch + 1, self.monitor, current, self.best)
                            )
            else:
                if self.verbose > 0:
                    lprint('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class StopCenterGradient2D(Callback):
    """
    Reset the weights of the center pixels after each training batch.
    """

    def __init__(self, blind_spots=None):
        super(StopCenterGradient2D, self).__init__()
        self.blind_spots = blind_spots

    def on_batch_ends(self, batch, logs=None):
        """
        At the end of each mini-batch, reset the gradient of center pixel to 0.
        """
        b = 0.001
        count = 0
        for lyr in self.model.layers:
            """
            This is to smooth the blind-spot kernel
            """
            if 'dilcv' in lyr.name and '_cv' in lyr.name and count < 3:
                """
                Get weights from the first 3 convolution layers
                """
                # Extract weights values from a layer
                weights = lyr.get_weights()  # (H, W, n_ch_in, n_ch_out or n_filters)

                # Get channel dim size
                num_channels = weights[0].shape[-2]  # n_ch_in

                # Create a smoothing kernel
                kernel = np.array([[b, b, b], [b, 1, b], [b, b, b]])

                if self.blind_spots:
                    for blind_spot in self.blind_spots:
                        spot_after_shift = tuple(map(sum, zip(blind_spot, (1, 1))))
                        kernel[spot_after_shift] = 0

                kernel = kernel[..., np.newaxis, np.newaxis].astype(np.float32)
                kernel /= kernel.sum()

                kernel = np.broadcast_to(
                    kernel, kernel.shape[:2] + (num_channels, num_channels)
                )

                # Apply smoothing kernel to the weights
                weights0 = backend.conv2d(
                    np.transpose(weights[0], [3, 0, 1, 2]),  # (batch, H, W, ch)
                    kernel,  # (H, W, n_ch_in (= n_ch of input tensor), n_ch_out)
                    padding='same',
                )

                # Plug the new weights values back to layer weight array
                weights[0] = np.transpose(weights0, [1, 2, 3, 0])

                # Set layer weights back to layer object
                lyr.set_weights(weights)

                # decrease for upper layers:
                b *= 0.1
                count += 1

        for lyr in self.model.layers:
            """
            This is to reset the gradient of the center of the kernel to 0. Only the first layer.
            """
            if 'dilcv0' in lyr.name and '_cv' in lyr.name:
                # Determine the index of center pixel
                indexes = tuple((i - 1) // 2 for i in lyr.kernel_size)

                # Extract weights values from a layer
                weights = lyr.get_weights()

                # Plug 0 to the center pixel of kernel weight
                weights[0][indexes[0], indexes[1], ...] = 0

                # Set layer weights back to layer object
                lyr.set_weight(weights)

    def on_epoch_end(self, epoch, logs=None):
        """
        This is the same process as self.on_batch_ends.
        To be precise, I'm actually not sure if I need this on epoch again on top of self.on_batch_ends.
        Just to be safe.
        """

        b = 0.001
        count = 0
        for lyr in self.model.layers:
            if 'dilcv' in lyr.name and '_cv' in lyr.name and count < 1:
                weights = lyr.get_weights()  # (H, W, n_ch_in, n_ch_out or n_filters)
                num_channels = weights[0].shape[-2]  # n_ch_in

                kernel = np.array([[b, b, b], [b, 1, b], [b, b, b]])

                if self.blind_spots:
                    for blind_spot in self.blind_spots:
                        spot_after_shift = tuple(map(sum, zip(blind_spot, (1, 1))))
                        kernel[spot_after_shift] = 0

                kernel = kernel[..., np.newaxis, np.newaxis].astype(np.float32)
                kernel /= kernel.sum()

                kernel = np.broadcast_to(
                    kernel, kernel.shape[:2] + (num_channels, num_channels)
                )
                weights0 = backend.conv2d(
                    np.transpose(weights[0], [3, 0, 1, 2]),  # (batch, H, W, ch)
                    kernel,  # (H, W, n_ch_in (= n_ch of input tensor), n_ch_out)
                    padding='same',
                )
                weights[0] = np.transpose(weights0, [1, 2, 3, 0])
                lyr.set_weights(weights)

                # decrease for upper layers:
                b *= 0.1
                count += 1

        for lyr in self.model.layers:
            if 'dilcv0' in lyr.name and '_cv' in lyr.name:
                indexes = tuple((i - 1) // 2 for i in lyr.kernel_size)
                weights = lyr.get_weights()
                weights[0][indexes[0], indexes[1], ...] = 0
                lyr.set_weights(weights)

    def on_train_end(self, logs=None):
        """
        This must be called after training to remove the blind-spot.
        This process is to slightly contaminate some of the original input information to the center pixel.
        So perhaps, a reverse action might be needed on_train_begin if we want to do retrain.
        """
        count = 0
        for lyr in self.model.layers:
            if 'dilcv' in lyr.name and '_cv' in lyr.name and count < 1:
                # Extract weights values from a layer
                weights = lyr.get_weights()  # (H, W, n_ch_in, n_ch_out or n_filters)

                # Get channel dim size
                num_channels = weights[0].shape[-2]  # n_ch_in

                # Create a smoothing kernel
                b = 1
                kernel = np.array([[b, b, b], [b, 1, b], [b, b, b]])

                if self.blind_spots:
                    for blind_spot in self.blind_spots:
                        spot_after_shift = tuple(map(sum, zip(blind_spot, (1, 1))))
                        kernel[spot_after_shift] = 0

                kernel = kernel[..., np.newaxis, np.newaxis].astype(np.float32)
                kernel /= kernel.sum()

                kernel = np.broadcast_to(
                    kernel, kernel.shape[:2] + (num_channels, num_channels)
                )

                # Apply smoothing kernel to the weights
                weights0 = backend.conv2d(
                    np.transpose(weights[0], [3, 0, 1, 2]),  # (batch, H, W, ch)
                    kernel,  # (H, W, n_ch_in (= n_ch of input tensor), n_ch_out)
                    padding='same',
                )

                # Plug the new weights values back to layer weight array
                weights0 = np.transpose(weights0, [1, 2, 3, 0])

                # Determine the index of cneter pixel
                indexes = tuple((i - 1) // 2 for i in lyr.kernel_size)

                # Plug 0 to the center pixel of kernel weight
                original_sum = weights[0].sum()
                a = weights0[indexes[0], indexes[1], ...]
                weights[0][indexes[0], indexes[1], ...] = a
                weights[0] *= original_sum / weights[0].sum()

                # Set layer weights back to layer object
                lyr.set_weights(weights)
                count += 1


class StopCenterGradient3D(Callback):
    """
    Reset the weights of the center pixels after each training batch.
    """

    def __init__(self, blind_spots=None):
        super(StopCenterGradient3D, self).__init__()
        self.blind_spots = blind_spots

    def on_batch_ends(self, batch, logs=None):
        """
        At the end of each mini-batch, reset the gradient of center pixel to 0.
        """
        b = 0.001
        count = 0
        for lyr in self.model.layers:
            """
            This is to smooth the blind-spot kernel
            """
            if 'dilcv' in lyr.name and '_cv' in lyr.name and count < 3:
                """
                Get weights from the first 3 convolution layers
                """
                # Extract weights values from a layer
                weights = lyr.get_weights()  # (H, W, n_ch_in, n_ch_out or n_filters)

                # Get channel dim size
                num_channels = weights[0].shape[-2]  # n_ch_in

                # Create a smoothing kernel
                kernel = np.array(
                    [
                        [[b, b, b], [b, b, b], [b, b, b]],
                        [[b, b, b], [b, 1, b], [b, b, b]],
                        [[b, b, b], [b, b, b], [b, b, b]],
                    ]
                )

                if self.blind_spots:
                    for blind_spot in self.blind_spots:
                        spot_after_shift = tuple(map(sum, zip(blind_spot, (1, 1, 1))))
                        kernel[spot_after_shift] = 0

                kernel = kernel[..., np.newaxis, np.newaxis].astype(np.float32)
                kernel /= kernel.sum()
                kernel = np.broadcast_to(
                    kernel, kernel.shape[:-2] + (num_channels, num_channels)
                )
                weights0 = backend.conv3d(
                    np.transpose(weights[0], [4, 0, 1, 2, 3]),  # (batch, D, H, W, ch)
                    kernel,  # (D, H, W, n_ch_in (= n_ch of input tensor), n_ch_out)
                    padding='same',
                )

                # Plug the new weights values back to layer weight array
                weights[0] = np.transpose(weights0, [1, 2, 3, 4, 0])

                # Set layer weights back to layer object
                lyr.set_weights(weights)

                # decrease for upper layers:
                b *= 0.1
                count += 1

        for lyr in self.model.layers:
            """
            This is to reset the gradient of the center of the kernel to 0. Only the first layer.
            """
            if 'dilcv0' in lyr.name and '_cv' in lyr.name:
                # Determine the index of center pixel
                indexes = tuple((i - 1) // 2 for i in lyr.kernel_size)

                # Extract weights values from a layer
                weights = lyr.get_weights()

                # Plug 0 to the center pixel of kernel weight
                weights[0][indexes[0], indexes[1], indexes[2], ...] = 0

                # Set layer weights back to layer object
                lyr.set_weight(weights)

    def on_epoch_end(self, epoch, logs=None):
        """
        This is the same process as self.on_batch_ends.
        To be precise, I'm actually not sure if I need this on epoch again on top of self.on_batch_ends.
        Just to be safe.
        """
        b = 0.001
        count = 0
        for lyr in self.model.layers:
            if 'dilcv' in lyr.name and '_cv' in lyr.name and count < 1:
                weights = lyr.get_weights()  # (H, W, n_ch_in, n_ch_out or n_filters)
                num_channels = weights[0].shape[-2]  # n_ch_in
                kernel = np.array(
                    [
                        [[b, b, b], [b, b, b], [b, b, b]],
                        [[b, b, b], [b, 1, b], [b, b, b]],
                        [[b, b, b], [b, b, b], [b, b, b]],
                    ]
                )

                if self.blind_spots:
                    for blind_spot in self.blind_spots:
                        spot_after_shift = tuple(map(sum, zip(blind_spot, (1, 1, 1))))
                        kernel[spot_after_shift] = 0

                kernel = kernel[..., np.newaxis, np.newaxis].astype(np.float32)
                kernel /= kernel.sum()
                kernel = np.broadcast_to(
                    kernel, kernel.shape[:-2] + (num_channels, num_channels)
                )
                weights0 = backend.conv3d(
                    np.transpose(weights[0], [4, 0, 1, 2, 3]),  # (batch, D, H, W, ch)
                    kernel,  # (D, H, W, n_ch_in (= n_ch of input tensor), n_ch_out)
                    padding='same',
                )
                weights[0] = np.transpose(weights0, [1, 2, 3, 4, 0])
                lyr.set_weights(weights)

                # decrease for upper layers:
                b *= 0.1
                count += 1

        for lyr in self.model.layers:
            if 'dilcv0' in lyr.name and '_cv' in lyr.name:
                indexes = tuple((i - 1) // 2 for i in lyr.kernel_size)
                weights = lyr.get_weights()
                weights[0][indexes[0], indexes[1], indexes[2], ...] = 0
                lyr.set_weights(weights)

    def on_train_end(self, logs=None):
        """
        This must be called after training to remove the blind-spot.
        This process is to slightly contaminate some of the original input information to the center pixel.
        So perhaps, a reverse action might be needed on_train_begin if we want to do retrain.
        """
        count = 0
        for lyr in self.model.layers:
            if 'dilcv' in lyr.name and '_cv' in lyr.name and count < 1:
                # Extract weights values from a layer
                weights = lyr.get_weights()  # (D, H, W, n_ch_in, n_ch_out or n_filters)

                # Get channel dim size
                num_channels = weights[0].shape[-2]  # n_ch_in

                # Create a smoothing kernel
                b = 1
                # kernel = np.array([[b, b, b], [b, 1, b], [b, b, b]])
                kernel = np.array(
                    [
                        [[b, b, b], [b, b, b], [b, b, b]],
                        [[b, b, b], [b, 1, b], [b, b, b]],
                        [[b, b, b], [b, b, b], [b, b, b]],
                    ]
                )

                if self.blind_spots:
                    for blind_spot in self.blind_spots:
                        spot_after_shift = tuple(map(sum, zip(blind_spot, (1, 1, 1))))
                        kernel[spot_after_shift] = 0

                kernel = kernel[..., np.newaxis, np.newaxis].astype(np.float32)
                kernel /= kernel.sum()
                kernel = np.broadcast_to(
                    kernel, kernel.shape[:-2] + (num_channels, num_channels)
                )

                # Apply smoothing kernel to the weights
                weights0 = backend.conv3d(
                    np.transpose(weights[0], [4, 0, 1, 2, 3]),  # (batch, D, H, W, ch)
                    kernel,  # (D, H, W, n_ch_in (= n_ch of input tensor), n_ch_out)
                    padding='same',
                )

                # Plug the new weights values back to layer weight array
                weights0 = np.transpose(weights0, [1, 2, 3, 4, 0])

                # Determine the index of cneter pixel
                indexes = tuple((i - 1) // 2 for i in lyr.kernel_size)

                # Plug 0 to the center pixel of kernel weight
                original_sum = weights[0].sum()
                a = weights0[indexes[0], indexes[1], indexes[2], ...]
                weights[0][indexes[0], indexes[1], indexes[2], ...] = a
                weights[0] *= original_sum / weights[0].sum()

                # Set layer weights back to layer object
                lyr.set_weights(weights)
                count += 1
