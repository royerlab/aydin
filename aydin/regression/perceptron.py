"""Multi-layer perceptron (neural network) regressor for Aydin's FGR pipeline.

This module provides :class:`PerceptronRegressor`, a Keras-based feed-forward
neural network regressor with early stopping, learning-rate scheduling, and
model checkpointing.
"""

import gc
import random
from os.path import exists, join

import keras
import numpy
import psutil
import tensorflow as tf
from keras.models import model_from_json
from keras.optimizers import Adam

from aydin.io.folders import get_temp_folder
from aydin.regression.base import RegressorBase
from aydin.regression.nn_utils.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    NNCallback,
    ReduceLROnPlateau,
)
from aydin.regression.nn_utils.models import feed_forward
from aydin.util.log.log import aprint, asection
from aydin.util.tf.device import get_best_device_name


class PerceptronRegressor(RegressorBase):
    """
    The Perceptron Regressor uses a simple multi-layer perceptron neural
    network. The big disadvantage of neural-network regressors is that they
    are trained stochastically, which usually means that when your run them
    twice you also get two different results. In some cases there can be
    significant variance between runs which can be problematic when trying
    to compare results.
    """

    device_max_mem = psutil.virtual_memory().total

    def __init__(
        self,
        max_epochs: int = 1024,
        learning_rate: float = 0.001,
        patience: int = 10,
        depth: int = 6,
        loss: str = 'l1',
    ):
        """Construct a multi-layer perceptron regressor.

        Parameters
        ----------
        max_epochs : int
            Maximum number of epochs allowed
        learning_rate : float
            Learning rate
            (advanced)
        patience : int
            Number of epochs required for early stopping
            (advanced)
        depth : int
            Depth of the model
        loss : str
            Type of loss to be used
            (advanced)

        """
        super().__init__()

        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.depth = depth

        loss = 'mae' if loss.lower() == 'l1' else loss
        loss = 'mse' if loss.lower() == 'l2' else loss
        self.loss = loss

        with asection("NN Regressor"):
            aprint("with no arguments")  # TODO: fix these logs

    def __repr__(self):
        return f"<{self.__class__.__name__}, max_epochs={self.max_epochs}, lr={self.learning_rate}, depth={self.depth}>"

    def _fit(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):
        """Fit a single-channel perceptron model.

        Trains a feed-forward neural network using Keras with early stopping
        and learning-rate reduction on plateau. Integer feature types are
        handled by rescaling targets to the integer range during training
        and inverting the scaling at prediction time.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training feature vectors of shape ``(n_samples, n_features)``.
        y_train : numpy.ndarray
            Training target values of shape ``(n_samples,)``.
        x_valid : numpy.ndarray, optional
            Validation feature vectors.
        y_valid : numpy.ndarray, optional
            Validation target values.
        regressor_callback : callable, optional
            Callback invoked at each training epoch.

        Returns
        -------
        _NNModel
            Fitted neural network model wrapper.
        """

        with asection("NN Regressor fitting:"):

            with tf.device(get_best_device_name()):
                # First we make sure that the arrays are of a type supported:
                def assert_type(array):
                    assert (
                        (array.dtype == numpy.float64)
                        or (array.dtype == numpy.float32)
                        or (array.dtype == numpy.float16)
                        or (array.dtype == numpy.uint16)
                        or (array.dtype == numpy.uint8)
                    )

                # Do we have a validation dataset?
                has_valid_dataset = x_valid is not None and y_valid is not None

                assert_type(x_train)
                assert_type(y_train)
                if has_valid_dataset:
                    assert_type(x_valid)
                    assert_type(y_valid)

                    # Types have to be consistent between train and valid sets:
                    assert x_train.dtype == x_valid.dtype
                    assert y_train.dtype == y_valid.dtype

                # In case the y dtype does not match the x dtype, we rescale and cast y:
                if numpy.issubdtype(x_train.dtype, numpy.integer) and numpy.issubdtype(
                    y_train.dtype, numpy.floating
                ):

                    # We remember the original type of y:
                    original_y_dtype = y_train.dtype

                    if x_train.dtype == numpy.uint8:
                        y_train *= 255
                        y_train = y_train.astype(numpy.uint8, copy=False)
                        if has_valid_dataset:
                            y_valid *= 255
                            y_valid = y_valid.astype(numpy.uint8, copy=False)
                        original_y_scale = 1 / 255.0

                    elif x_train.dtype == numpy.uint16:
                        y_train *= 255 * 255
                        y_train = y_train.astype(numpy.uint16, copy=False)
                        if has_valid_dataset:
                            y_valid *= 255 * 255
                            y_valid = y_valid.astype(numpy.uint16, copy=False)
                        original_y_scale = 1 / (255.0 * 255.0)
                else:
                    original_y_dtype = None
                    original_y_scale = None

                # Get the number of entries and features from the array shape:
                nb_data_points = x_train.shape[0]
                num_features = x_train.shape[-1]

                aprint(f"Number of data points : {nb_data_points}")
                if has_valid_dataset:
                    aprint(f"Number of validation data points: {x_valid.shape[0]}")
                aprint(f"Number of features per data point: {num_features}")

                # Shapes of both x and y arrays:
                x_shape = (-1, num_features)
                y_shape = (-1, 1)

                # Learning rate and decay:
                learning_rate = self.learning_rate
                learning_rate_decay = 0.1 * self.learning_rate
                aprint(f"Learning rate: {learning_rate}")
                aprint(f"Learning rate decay: {learning_rate_decay}")

                # Weight decay and noise:
                weight_decay = 0.01 * self.learning_rate
                noise = 0.1 * self.learning_rate
                aprint(f"Weight decay: {weight_decay}")
                aprint(f"Added noise: {noise}")

                # Initialise model if not done yet:
                model = feed_forward(
                    num_features,
                    depth=self.depth,
                    weight_decay=weight_decay,
                    noise=noise,
                )
                opt = Adam(learning_rate=learning_rate)
                model.compile(optimizer=opt, loss=self.loss)

                aprint(f"Number of parameters in model: {model.count_params()}")

                # Reshape arrays:
                x_train = x_train.reshape(x_shape)
                y_train = y_train.reshape(y_shape)

                if x_valid is not None and y_valid is not None:
                    x_valid = x_valid.reshape(x_shape)
                    y_valid = y_valid.reshape(y_shape)

                batch_size = 1024
                aprint(f"Keras batch size for training: {batch_size}")

                # Effective number of epochs:
                effective_number_of_epochs = self.max_epochs
                aprint(f"Effective max number of epochs: {effective_number_of_epochs}")

                # Early stopping patience:
                early_stopping_patience = self.patience
                aprint(f"Early stopping patience: {early_stopping_patience}")

                # Effective LR patience:
                effective_lr_patience = max(1, self.patience // 2)
                aprint(f"Effective LR patience: {effective_lr_patience}")

                # Here is the list of callbacks:
                callbacks = []

                # Set upstream callback:
                keras_callback = NNCallback(regressor_callback)

                # Early stopping callback:
                early_stopping = EarlyStopping(
                    self,
                    monitor='val_loss',
                    min_delta=min(0.000001, 0.1 * self.learning_rate),
                    patience=early_stopping_patience,
                    mode='auto',
                    restore_best_weights=True,
                )

                # Reduce LR on plateau:
                reduce_learning_rate = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    verbose=1,
                    patience=effective_lr_patience,
                    mode='auto',
                    min_lr=0.0001 * self.learning_rate,
                )

                model_file_path = join(
                    get_temp_folder(),
                    f"aydin_nn_keras_model_file_{random.randint(0, 1e16)}.hdf5",
                )
                checkpoint = ModelCheckpoint(
                    model_file_path, monitor='val_loss', verbose=1, save_best_only=True
                )

                # Add callbacks to the list:
                callbacks.append(keras_callback)
                callbacks.append(early_stopping)
                callbacks.append(reduce_learning_rate)
                callbacks.append(checkpoint)

                # x_train = x_train.astype(numpy.float64)
                # y_train = y_train.astype(numpy.float64)
                # x_valid = x_valid.astype(numpy.float64)
                # y_valid = y_valid.astype(numpy.float64)

                # Training happens here:
                with asection("NN regressor fitting now:"):
                    train_history = model.fit(
                        x_train,
                        y_train,
                        validation_data=(
                            (x_valid, y_valid)
                            if (x_valid is not None and y_valid is not None)
                            else None
                        ),
                        epochs=effective_number_of_epochs,
                        batch_size=min(batch_size, nb_data_points),
                        shuffle=True,
                        verbose=0,  # 0 if is_batch else 1,
                        callbacks=callbacks,
                    )
                    aprint("NN regressor fitting done.")

                del x_train
                del y_train

                # Reload the best weights:
                if exists(model_file_path):
                    aprint("Loading best model to date.")
                    model.load_weights(model_file_path)

                # loss_history = train_history.history['loss']
                # aprint(f"Loss history after training: {loss_history}")

                if 'val_loss' in train_history.history:
                    self.last_val_loss = train_history.history['val_loss'][0]

                loss_history = {
                    'training': train_history.history['loss'],
                    'validation': train_history.history['val_loss'],
                }

                gc.collect()

                return _NNModel(model, original_y_dtype, original_y_scale, loss_history)


class _NNModel:
    """Internal wrapper around a fitted Keras model.

    Handles serialisation, deserialisation, prediction, and optional
    dtype/scale inversion for a single output channel.

    Attributes
    ----------
    model : keras.Model
        The underlying Keras model.
    original_y_dtype : numpy.dtype or None
        Original target dtype before integer rescaling, or ``None`` if no
        rescaling was applied.
    original_y_scale : float or None
        Scale factor to invert integer rescaling at prediction time.
    loss_history : dict
        Training and validation loss arrays.
    """

    def __init__(self, model, original_y_dtype, original_y_scale, loss_history):
        self.model = model
        self.original_y_dtype = original_y_dtype
        self.original_y_scale = original_y_scale
        self.loss_history = loss_history

    def _save_internals(self, path: str):
        """Save the Keras model architecture and weights to the given directory.

        Parameters
        ----------
        path : str
            Directory in which to write ``keras_model.txt`` (architecture)
            and ``keras_weights.txt`` (weights).
        """

        if self.model is not None:
            # serialize model to JSON:
            keras_model_file = join(path, 'keras_model.txt')
            model_json = self.model.to_json()
            with open(keras_model_file, "w") as json_file:
                json_file.write(model_json)

            # serialize weights to HDF5:
            keras_model_file = join(path, 'keras_weights.txt')
            self.model.save_weights(keras_model_file)

    def _load_internals(self, path: str):
        """Load the Keras model architecture and weights from the given directory.

        Parameters
        ----------
        path : str
            Directory containing ``keras_model.txt`` and ``keras_weights.txt``.
        """
        # load JSON and create model:
        keras_model_file = join(path, 'keras_model.txt')
        with open(keras_model_file, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.model = model_from_json(loaded_model_json)
        # load weights into new model:
        keras_model_file = join(path, 'keras_weights.txt')
        self.model.load_weights(keras_model_file).expect_partial()

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        return state

    def predict(self, x):
        """Predict target values for the given feature vectors.

        Automatically batches the prediction to fit within available memory
        and inverts any integer-range rescaling that was applied during
        training.

        Parameters
        ----------
        x : numpy.ndarray
            Feature vectors of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Predicted values, cast back to the original target dtype if
            integer rescaling was used during training.
        """
        with asection("NN Regressor prediction:"):

            with tf.device(get_best_device_name()):
                aprint(f"Number of data points             : {x.shape[0]}")
                aprint(f"Number of features per data points: {x.shape[-1]}")

                # Number of features:
                num_of_features = x.shape[-1]

                # We check that we get the right number of features.
                # If not, most likely the batch_dims are set wrong...
                assert num_of_features == x.shape[-1]

                # How much memory is available in GPU:
                max_gpu_mem_in_bytes = PerceptronRegressor.device_max_mem

                # We limit ourselves to using only a quarter of GPU memory:
                max_number_of_floats = (max_gpu_mem_in_bytes // 4) // 4

                # Max size of batch:
                max_gpu_batch_size = max_number_of_floats / num_of_features

                # Batch size taking all this into account:
                batch_size = max(1, min(max_gpu_batch_size, x.shape[0] // 256))

                # Heuristic threshold here obtained by inspecting batch size per GPU memory
                # Basically ensures ratio of 700000 batch size per 12GBs of GPU memory
                batch_size = min(
                    batch_size, (700000 * max_gpu_mem_in_bytes) // 12884901888
                )

                aprint(f"Batch size: {batch_size}")
                aprint(f"Predicting. features shape = {x.shape}")

                aprint("NN regressor predicting now...")
                yp = self.model.predict(x, batch_size=batch_size)
                aprint("NN regressor predicting done!")

                # We cast back yp to the correct type and range:
                if self.original_y_dtype is not None:
                    yp = yp.astype(self.original_y_dtype)
                    yp *= self.original_y_scale

                return yp
