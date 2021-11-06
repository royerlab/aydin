import gc
import os
from abc import ABC, abstractmethod
from os.path import join

import jsonpickle
import numpy

from aydin.util.misc.json import encode_indent
from aydin.util.log.log import lprint


class RegressorBase(ABC):
    """Regressor base class"""

    def __init__(self):
        super().__init__()
        self._stop_fit = False
        self.num_channels = None
        self.models = 0

        self.loss_history = []

    def recommended_max_num_datapoints(self) -> int:
        """Recommended maximum number of datapoints

        Returns
        -------
        int

        """
        # very large number, essentially no limit by default
        return 1e9

    @abstractmethod
    def _fit(self, x_train, y_train, x_valid, y_valid, regressor_callback=None):
        """Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).

        Parameters
        ----------
        x_train
            x training values
        y_train
            y training values
        x_valid
            x validation values
        y_valid
            y validation values
        regressor_callback
            regressor callback

        """

    def fit(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):
        """Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).
        The target y_train can have multiple 'channels'. This will cause multiple regressors
        to be instanciated internally to be able to predict these channels from the input features.

        Parameters
        ----------
        x_train
            x training values
        y_train
            y training values
        x_valid
            x validation values
        y_valid
            y validation values
        regressor_callback
            regressor callback

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
            self.loss_history.append(model_channel.loss_history)

        self.num_channels = len(self.models)

    def predict(self, x, models_to_use=None):
        """Predicts y given x by applying the learned function f: y=f(x)
        If the regressor is trained on multiple ouput channels, this will
        return the corresponding number of channels...

        Parameters
        ----------
        x
            x values
        models_to_use

        Returns
        -------
        numpy.typing.ArrayLike
            inferred y values

        """
        if models_to_use is None:
            models_to_use = self.models

        return numpy.stack([model.predict(x) for model in models_to_use])

    def stop_fit(self):
        """Stops training (can be called by another thread)"""
        self._stop_fit = True

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['models']
        return state

    def save(self, path: str):
        """Saves an 'all-batteries-included' regressor at a given path (folder).

        Parameters
        ----------
        path
            path to save to

        Returns
        -------
        frozen : Encoded JSON object

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
        """Returns an 'all-batteries-included' regressor from a given path (folder).

        Parameters
        ----------
        path
            path to load from.

        Returns
        -------
        thawed : object

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
