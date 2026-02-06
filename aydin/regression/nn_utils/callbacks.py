"""Keras training callbacks for the perceptron regressor.

Provides custom Keras callbacks for early stopping, learning-rate reduction,
model checkpointing, and integration with Aydin's regressor callback interface.
"""

import warnings

import keras
import numpy as np
from keras.callbacks import Callback

from aydin.util.log.log import lprint


class NNCallback(Callback):
    """Keras callback that bridges to Aydin's image translator callback interface.

    Forwards epoch-end and training-end events to an upstream regressor
    callback function with signature ``(iteration, val_loss, model)``.

    Parameters
    ----------
    regressor_callback : callable or None
        Upstream callback to notify. If ``None``, notifications are silently
        skipped.
    """

    def __init__(self, regressor_callback):
        super().__init__()

        self.regressor_callback = regressor_callback
        self.iteration = 0

    def on_train_begin(self, logs=None):
        """Called at the beginning of training (no-op)."""
        pass

    def on_batch_ends(self, batch, logs=None):
        """Called at the end of each batch; forwards to :meth:`notify`."""
        self.notify(logs)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch; forwards to :meth:`notify`."""
        self.notify(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training; forwards to :meth:`notify`."""
        self.notify(logs)

    def notify(self, logs):
        """Forward current training state to the upstream regressor callback.

        Parameters
        ----------
        logs : dict or None
            Keras training logs for the current event.
        """
        if self.regressor_callback:
            iteration = self.iteration
            val_loss = self.get_monitor_value(logs)
            model = self.model
            self.regressor_callback(iteration, val_loss, model)
            self.iteration += 1

    def get_monitor_value(self, logs):
        """Extract the validation loss from training logs.

        Parameters
        ----------
        logs : dict or None
            Keras training logs.

        Returns
        -------
        float or None
            The ``'val_loss'`` value, or ``None`` if unavailable.
        """
        if logs is not None:
            monitor_value = logs.get('val_loss')
            return monitor_value
        return None


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.

    Also integrates with Aydin's external stop mechanism: training is
    halted if ``nn_regressor._stop_fit`` becomes ``True``.

    Parameters
    ----------
    nn_regressor : PerceptronRegressor
        Parent regressor instance whose ``_stop_fit`` flag is checked
        each epoch.
    monitor : str
        Name of the metric to monitor (e.g. ``'val_loss'``).
    min_delta : float
        Minimum change in the monitored metric to qualify as an
        improvement.
    patience : int
        Number of epochs with no improvement after which training stops.
    mode : str
        One of ``{'auto', 'min', 'max'}``. Determines whether improvement
        means decreasing or increasing the monitored metric.
    baseline : float, optional
        Baseline value the monitored metric must beat.
    restore_best_weights : bool
        If ``True``, restore model weights from the epoch with the best
        monitored metric value.
    """

    def __init__(
        self,
        nn_regressor,
        monitor='val_loss',
        min_delta=0,
        patience=0,
        mode='auto',
        baseline=None,
        restore_best_weights=False,
    ):
        super(EarlyStopping, self).__init__()

        self.nn_regressor = nn_regressor
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
        """Reset internal state to allow callback re-use."""
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.inf if self.monitor_op == np.less else -np.inf

    def on_epoch_end(self, epoch, logs=None):
        """Check for improvement and optionally stop training.

        Parameters
        ----------
        epoch : int
            Current epoch index.
        logs : dict or None
            Keras training logs.
        """
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

        # This is where we stop training:
        if self.nn_regressor._stop_fit:
            lprint('Training externally stopped!')
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        """Log the early stopping epoch if applicable."""
        if self.stopped_epoch > 0:
            lprint('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        """Extract the monitored metric value from training logs.

        Parameters
        ----------
        logs : dict
            Keras training logs.

        Returns
        -------
        float or None
            The monitored metric value, or ``None`` if unavailable.
        """
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
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2--10 once learning stagnates. This callback monitors a quantity
    and if no improvement is seen for a ``patience`` number of epochs,
    the learning rate is reduced.

    Parameters
    ----------
    monitor : str
        Name of the metric to monitor (e.g. ``'val_loss'``).
    factor : float
        Factor by which the learning rate is reduced:
        ``new_lr = lr * factor``. Must be less than 1.0.
    patience : int
        Number of epochs with no improvement after which the learning
        rate is reduced.
    verbose : int
        Verbosity mode. ``0``: quiet, ``1``: log update messages.
    mode : str
        One of ``{'auto', 'min', 'max'}``. Determines whether improvement
        means decreasing or increasing the monitored metric.
    min_delta : float
        Threshold for measuring a new optimum; only changes larger than
        ``min_delta`` count as improvements.
    cooldown : int
        Number of epochs to wait before resuming normal operation after
        the learning rate has been reduced.
    min_lr : float
        Lower bound on the learning rate.
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
            self.best = np.inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        """Reset internal state at the start of training."""
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        """Check for improvement and optionally reduce learning rate.

        Parameters
        ----------
        epoch : int
            Current epoch index.
        logs : dict or None
            Keras training logs.
        """
        logs = logs or {}
        logs['lr'] = float(self.model.optimizer.learning_rate)
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
                    old_lr = float(self.model.optimizer.learning_rate)
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.model.optimizer.learning_rate.assign(new_lr)
                        if self.verbose > 0:
                            lprint(
                                'Epoch %05d: ReduceLROnPlateau reducing '
                                'learning rate to %s.' % (epoch + 1, new_lr)
                            )
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        """Return whether the callback is currently in a cooldown period.

        Returns
        -------
        bool
            ``True`` if cooldown is active.
        """
        return self.cooldown_counter > 0


class ModelCheckpoint(Callback):
    """Save the model after every epoch (or periodically).

    ``filepath`` can contain named formatting options that are filled with
    the value of ``epoch`` and keys in ``logs`` (passed in ``on_epoch_end``).
    For example, ``'weights.{epoch:02d}-{val_loss:.2f}.hdf5'`` will include
    the epoch number and validation loss in the filename.

    Parameters
    ----------
    filepath : str
        Path (optionally with format placeholders) to save the model file.
    monitor : str
        Metric to monitor for ``save_best_only`` mode.
    verbose : int
        Verbosity mode: ``0`` (quiet) or ``1`` (log messages).
    save_best_only : bool
        If ``True``, only overwrite the saved model when the monitored
        metric improves.
    save_weights_only : bool
        If ``True``, save only the model weights; otherwise save the full
        model.
    mode : str
        One of ``{'auto', 'min', 'max'}``. Determines the direction of
        improvement for the monitored metric.
    period : int
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
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.inf
            else:
                self.monitor_op = np.less
                self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        """Save the model if checkpoint criteria are met.

        Parameters
        ----------
        epoch : int
            Current epoch index.
        logs : dict or None
            Keras training logs.
        """
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    lprint(
                        f'Warning: Can save best model only with {self.monitor} available, skipping.'
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            lprint(
                                'Epoch %05d: %s=%0.5f improved from %0.5f to %0.5f (saving model),'
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
