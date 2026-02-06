"""Base class for all regression methods used in Aydin's FGR pipeline.

This module defines the abstract base class :class:`RegressorBase` that every
regressor backend must implement. It provides multi-channel training support,
serialisation/deserialisation, and a common prediction interface.
"""

import gc
import os
from abc import ABC, abstractmethod
from os.path import join

import jsonpickle
import numpy

from aydin.util.log.log import lprint
from aydin.util.misc.json import encode_indent


class RegressorBase(ABC):
    """Abstract base class for all Aydin regressors.

    Provides common infrastructure for fitting a function ``y = f(x)`` from
    feature vectors to target values, including multi-channel support (one
    internal model per output channel), early-stopping hooks, and
    JSON-based serialisation.

    Attributes
    ----------
    num_channels : int or None
        Number of output channels after training.
    models : list
        List of per-channel fitted model objects.
    loss_history : list
        Per-channel loss history collected during training.
    """

    def __init__(self):
        super().__init__()
        self._stop_fit = False
        self.num_channels = None
        self.models = 0

        self.loss_history = []

    def recommended_max_num_datapoints(self) -> int:
        """Return the recommended maximum number of data points for this regressor.

        Subclasses may override this to reflect hardware or algorithmic
        constraints (e.g. GPU memory limits).

        Returns
        -------
        int
            Upper bound on the number of data points to use for training.
        """
        # very large number, essentially no limit by default
        return int(1e9)

    @abstractmethod
    def _fit(self, x_train, y_train, x_valid, y_valid, regressor_callback=None):
        """Fit function y=f(x) given training pairs (x_train, y_train).

        This is the internal fitting method that each subclass must implement
        for a single output channel. Training should stop when performance
        ceases to improve on the validation dataset (x_valid, y_valid).

        Parameters
        ----------
        x_train : numpy.ndarray
            Training feature vectors of shape ``(n_samples, n_features)``.
        y_train : numpy.ndarray
            Training target values of shape ``(n_samples,)``.
        x_valid : numpy.ndarray
            Validation feature vectors.
        y_valid : numpy.ndarray
            Validation target values.
        regressor_callback : callable, optional
            Callback invoked at each training iteration with signature
            ``(iteration, val_loss, model)``.

        Returns
        -------
        object
            A fitted model object that exposes a ``predict(x)`` method.
        """

    def fit(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):
        """Fit function y=f(x) given training pairs (x_train, y_train).

        Training stops when performance ceases to improve on the validation
        dataset (x_valid, y_valid). If ``y_train`` has multiple channels
        (i.e. ``y_train.ndim > 1``), one internal regressor model is
        instantiated per channel.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training feature vectors of shape ``(n_samples, n_features)``.
        y_train : numpy.ndarray
            Training target values. Shape ``(n_samples,)`` for single-channel
            or ``(n_channels, n_samples)`` for multi-channel.
        x_valid : numpy.ndarray, optional
            Validation feature vectors. Defaults to ``x_train`` if not provided.
        y_valid : numpy.ndarray, optional
            Validation target values. Defaults to ``y_train`` if not provided.
        regressor_callback : callable, optional
            Callback invoked at each training iteration with signature
            ``(iteration, val_loss, model)``.
        """
        has_more_than_one_channel = len(y_train.shape) > 1

        if x_valid is None:
            x_valid = x_train
            y_valid = y_train

        # to the multi-channel form, but with just one chanel;
        if not has_more_than_one_channel:
            y_train = y_train[numpy.newaxis, ...]
            y_valid = y_valid[numpy.newaxis, ...]

        self.models = []
        self._stop_fit = False
        for y_train_channel, y_valid_channel in zip(y_train, y_valid):

            gc.collect()
            model_channel = self._fit(
                x_train, y_train_channel, x_valid, y_valid_channel, regressor_callback
            )
            self.models.append(model_channel)
            if hasattr(model_channel, 'loss_history'):
                self.loss_history.append(model_channel.loss_history)

        self.num_channels = len(self.models)

    def predict(self, x, models_to_use=None):
        """Predict y given x by applying the learned function f: y = f(x).

        If the regressor was trained on multiple output channels, the result
        is stacked along a new leading axis so that ``result.shape[0]``
        equals the number of channels.

        Parameters
        ----------
        x : numpy.ndarray
            Feature vectors of shape ``(n_samples, n_features)``.
        models_to_use : list, optional
            Subset of internal models to use for prediction. If ``None``,
            all trained models are used.

        Returns
        -------
        numpy.ndarray
            Predicted target values.
        """
        if models_to_use is None:
            models_to_use = self.models

        return numpy.stack([model.predict(x) for model in models_to_use])

    def stop_fit(self):
        """Request an early stop of the current training run.

        This method is thread-safe and can be called from a different thread
        to interrupt an ongoing ``fit`` call.
        """
        self._stop_fit = True

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['models']
        return state

    def save(self, path: str):
        """Save the regressor and all its internal models to a folder.

        The regressor metadata is stored as ``regressor.json`` and each
        per-channel model is stored in a ``channel<i>/`` sub-folder.

        Parameters
        ----------
        path : str
            Directory path where the regressor will be saved. Created if
            it does not already exist.

        Returns
        -------
        str
            JSON-encoded representation of the regressor.
        """
        os.makedirs(path, exist_ok=True)

        frozen = encode_indent(self)

        lprint(f"Saving regressor to: {path}")
        with open(join(path, "regressor.json"), "w") as json_file:
            json_file.write(frozen)

        for i, model in enumerate(self.models):
            channel_path = join(path, f"channel{i}")
            os.makedirs(channel_path, exist_ok=True)
            frozen_model = encode_indent(model)
            with open(join(channel_path, "regressor_model.json"), "w") as json_file:
                json_file.write(frozen_model)

            model._save_internals(channel_path)

        return frozen

    @staticmethod
    def load(path: str):
        """Load a previously saved regressor from a folder.

        Parameters
        ----------
        path : str
            Directory path that was used during :meth:`save`.

        Returns
        -------
        RegressorBase
            The deserialised regressor instance with all models restored.
        """
        lprint(f"Loading regressor from: {path}")
        with open(join(path, "regressor.json"), "r") as json_file:
            frozen = json_file.read()

        thawed = jsonpickle.decode(frozen)

        thawed.models = []
        for i in range(thawed.num_channels):
            channel_path = join(path, f"channel{i}")
            lprint(f"Loading regressor model for channel {i} from: {path}")
            with open(join(channel_path, "regressor_model.json"), "r") as json_file:
                frozen_model = json_file.read()

            thawed_model = jsonpickle.decode(frozen_model)
            thawed_model._load_internals(channel_path)
            thawed.models.append(thawed_model)

        return thawed
