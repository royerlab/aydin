"""Multi-layer perceptron (neural network) regressor for Aydin's FGR pipeline.

This module provides :class:`PerceptronRegressor`, a PyTorch-based feed-forward
neural network regressor with early stopping, learning-rate scheduling, and
best-model checkpointing.
"""

import gc
import json
import math
from collections import OrderedDict
from os.path import join

import numpy
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from aydin.regression.base import RegressorBase
from aydin.regression.nn_utils.models_torch import FeedForwardModel
from aydin.util.log.log import aprint, asection
from aydin.util.torch.device import available_device_memory, get_torch_device


class PerceptronRegressor(RegressorBase):
    """Multi-layer perceptron (MLP) neural network regressor.

    Uses a PyTorch-based feed-forward neural network with residual connections,
    early stopping, learning-rate scheduling, and best-model checkpointing. The
    big disadvantage of neural-network regressors is that they are trained
    stochastically, which usually means that running them twice produces
    two different results. In some cases there can be significant variance
    between runs, which can be problematic when trying to compare results.
    <notgui>
    """

    @staticmethod
    def _get_device_max_mem():
        """Return available device memory on demand (not at import time)."""
        return available_device_memory()

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
            Maximum number of training epochs allowed.
        learning_rate : float
            Initial learning rate for the Adam optimiser.
            (advanced)
        patience : int
            Number of epochs without improvement on the validation loss
            before early stopping is triggered.
            (advanced)
        depth : int
            Depth of the feed-forward network (number of dense blocks).
        loss : str
            Loss function name. Accepts ``'l1'`` (mapped to MAE),
            ``'l2'`` (mapped to MSE), or ``'mae'``/``'mse'``.
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
        self.last_val_loss = None

        with asection("NN Regressor"):
            aprint("with no arguments")  # TODO: fix these logs

    def __repr__(self):
        """Return a concise string representation of the regressor."""
        return f"<{self.__class__.__name__}, max_epochs={self.max_epochs}, lr={self.learning_rate}, depth={self.depth}>"

    def _fit(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):
        """Fit a single-channel perceptron model.

        Trains a feed-forward neural network using PyTorch with early stopping
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

            device = get_torch_device()

            # First we make sure that the arrays are of a type supported:
            def assert_type(array):
                """Assert that the array dtype is supported by the perceptron.

                Supported dtypes are float64, float32, float16, uint16,
                and uint8.

                Parameters
                ----------
                array : numpy.ndarray
                    Array whose dtype is validated.

                Raises
                ------
                AssertionError
                    If the array dtype is not one of the supported types.
                """
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
                    y_train = y_train * 255
                    y_train = y_train.astype(numpy.uint8, copy=False)
                    if has_valid_dataset:
                        y_valid = y_valid * 255
                        y_valid = y_valid.astype(numpy.uint8, copy=False)
                    original_y_scale = 1 / 255.0

                elif x_train.dtype == numpy.uint16:
                    y_train = y_train * 65535
                    y_train = y_train.astype(numpy.uint16, copy=False)
                    if has_valid_dataset:
                        y_valid = y_valid * 65535
                        y_valid = y_valid.astype(numpy.uint16, copy=False)
                    original_y_scale = 1 / 65535.0
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
            aprint(f"Learning rate: {learning_rate}")

            # Weight decay and noise:
            weight_decay = 0.01 * self.learning_rate
            noise = 0.1 * self.learning_rate
            aprint(f"Weight decay: {weight_decay}")
            aprint(f"Added noise: {noise}")

            # Initialise model:
            model = FeedForwardModel(
                num_features,
                depth=self.depth,
                weight_decay=weight_decay,
                noise=noise,
            )
            model = model.to(device)

            num_params = sum(p.numel() for p in model.parameters())
            aprint(f"Number of parameters in model: {num_params}")

            # Reshape arrays:
            x_train = x_train.reshape(x_shape).astype(numpy.float32)
            y_train = y_train.reshape(y_shape).astype(numpy.float32)

            if has_valid_dataset:
                x_valid = x_valid.reshape(x_shape).astype(numpy.float32)
                y_valid = y_valid.reshape(y_shape).astype(numpy.float32)

            batch_size = 1024
            aprint(f"Batch size for training: {batch_size}")

            # Effective number of epochs:
            effective_number_of_epochs = self.max_epochs
            aprint(f"Effective max number of epochs: {effective_number_of_epochs}")

            # Early stopping patience:
            early_stopping_patience = self.patience
            aprint(f"Early stopping patience: {early_stopping_patience}")

            # Effective LR patience:
            effective_lr_patience = max(1, self.patience // 2)
            aprint(f"Effective LR patience: {effective_lr_patience}")

            # Loss function
            if self.loss in ('mae', 'l1'):
                loss_fn = torch.nn.L1Loss()
            else:
                loss_fn = torch.nn.MSELoss()

            optimizer = Adam(model.parameters(), lr=learning_rate, eps=1e-7)
            scheduler = ReduceLROnPlateau(
                optimizer,
                'min',
                factor=0.5,
                patience=effective_lr_patience,
                min_lr=0.0001 * self.learning_rate,
            )

            # Create data loaders
            train_dataset = TensorDataset(
                torch.from_numpy(x_train), torch.from_numpy(y_train)
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=min(batch_size, nb_data_points),
                shuffle=True,
                num_workers=0,
            )

            valid_loader = None
            if has_valid_dataset:
                valid_dataset = TensorDataset(
                    torch.from_numpy(x_valid), torch.from_numpy(y_valid)
                )
                valid_loader = DataLoader(
                    valid_dataset,
                    batch_size=min(batch_size, x_valid.shape[0]),
                    shuffle=False,
                    num_workers=0,
                )

            # Training loop
            best_val_loss = math.inf
            patience_counter = 0
            best_model_state_dict = None
            loss_history = {'training': [], 'validation': []}

            with asection("NN regressor fitting now:"):
                for epoch in range(effective_number_of_epochs):
                    # Training phase
                    model.train()
                    train_loss_accum = 0.0
                    train_batches = 0
                    for x_batch, y_batch in train_loader:
                        x_batch = x_batch.to(device)
                        y_batch = y_batch.to(device)

                        optimizer.zero_grad()
                        pred = model(x_batch)
                        loss = loss_fn(pred, y_batch) + model.l1_penalty()
                        loss.backward()
                        optimizer.step()

                        train_loss_accum += loss.item()
                        train_batches += 1

                    avg_train_loss = train_loss_accum / max(train_batches, 1)
                    loss_history['training'].append(avg_train_loss)

                    # Validation phase
                    avg_val_loss = avg_train_loss  # fallback if no validation set
                    if valid_loader is not None:
                        model.eval()
                        val_loss_accum = 0.0
                        val_batches = 0
                        with torch.no_grad():
                            for x_batch, y_batch in valid_loader:
                                x_batch = x_batch.to(device)
                                y_batch = y_batch.to(device)
                                pred = model(x_batch)
                                val_loss = loss_fn(pred, y_batch)
                                val_loss_accum += val_loss.item()
                                val_batches += 1
                        avg_val_loss = val_loss_accum / max(val_batches, 1)

                    loss_history['validation'].append(avg_val_loss)

                    scheduler.step(avg_val_loss)

                    # Callback
                    if regressor_callback is not None:
                        regressor_callback(epoch, avg_val_loss, model)

                    # Early stopping
                    min_delta = min(0.000001, 0.1 * self.learning_rate)
                    if avg_val_loss < best_val_loss - min_delta:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        best_model_state_dict = OrderedDict(
                            {k: v.to('cpu') for k, v in model.state_dict().items()}
                        )
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            aprint(f"Early stopping at epoch {epoch + 1}")
                            break

                    # External stop
                    if self._stop_fit:
                        aprint('Training externally stopped!')
                        break

                aprint("NN regressor fitting done.")

            # Restore best weights
            if best_model_state_dict is not None:
                model.load_state_dict(best_model_state_dict)
                model = model.to(device)

            del x_train
            del y_train

            if has_valid_dataset and len(loss_history['validation']) > 0:
                self.last_val_loss = best_val_loss

            gc.collect()

            return _NNModel(
                model, original_y_dtype, original_y_scale, loss_history, device
            )


class _NNModel:
    """Internal wrapper around a fitted PyTorch model.

    Handles serialisation, deserialisation, prediction, and optional
    dtype/scale inversion for a single output channel.

    Attributes
    ----------
    model : torch.nn.Module
        The underlying PyTorch model.
    original_y_dtype : numpy.dtype or None
        Original target dtype before integer rescaling, or ``None`` if no
        rescaling was applied.
    original_y_scale : float or None
        Scale factor to invert integer rescaling at prediction time.
    loss_history : dict
        Training and validation loss arrays.
    device : torch.device
        Device on which the model runs.
    """

    def __init__(
        self, model, original_y_dtype, original_y_scale, loss_history, device=None
    ):
        """Initialise the PyTorch model wrapper.

        Parameters
        ----------
        model : torch.nn.Module
            Trained PyTorch model.
        original_y_dtype : numpy.dtype or None
            Original target dtype before integer rescaling, or ``None``
            if no rescaling was applied.
        original_y_scale : float or None
            Scale factor to invert integer rescaling at prediction time.
        loss_history : dict
            Dictionary with ``'training'`` and ``'validation'`` loss lists.
        device : torch.device, optional
            Device for inference. If ``None``, uses CPU.
        """
        self.model = model
        self.original_y_dtype = original_y_dtype
        self.original_y_scale = original_y_scale
        self.loss_history = loss_history
        self.device = device or torch.device('cpu')

    def _save_internals(self, path: str):
        """Save the PyTorch model architecture metadata and weights.

        Parameters
        ----------
        path : str
            Directory in which to write ``torch_nn_metadata.json``
            and ``torch_nn_weights.pth``.
        """
        if self.model is not None:
            # Save metadata
            metadata = {
                'n_features': self.model.n_features,
                'depth': self.model.depth,
                'noise_std': self.model.noise_std,
                'weight_decay': self.model.weight_decay,
            }
            with open(join(path, 'torch_nn_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save weights
            torch.save(self.model.state_dict(), join(path, 'torch_nn_weights.pth'))

    def _load_internals(self, path: str):
        """Load the PyTorch model from the given directory.

        Parameters
        ----------
        path : str
            Directory containing ``torch_nn_metadata.json`` and
            ``torch_nn_weights.pth``.
        """
        with open(join(path, 'torch_nn_metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.model = FeedForwardModel(
            n_features=metadata['n_features'],
            depth=metadata['depth'],
            noise=metadata['noise_std'],
            weight_decay=metadata['weight_decay'],
        )
        self.model.load_state_dict(
            torch.load(join(path, 'torch_nn_weights.pth'), weights_only=True)
        )
        self.model.eval()
        self.device = torch.device('cpu')

    def __getstate__(self):
        """Return pickling state, excluding the non-serialisable PyTorch model.

        Returns
        -------
        dict
            Instance state with the ``model`` and ``device`` keys removed.
        """
        state = self.__dict__.copy()
        del state['model']
        del state['device']
        return state

    def __setstate__(self, state):
        """Restore pickling state with safe defaults for excluded attributes.

        Parameters
        ----------
        state : dict
            Pickled state (without ``model`` and ``device``).
        """
        self.__dict__.update(state)
        self.model = None
        self.device = None

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

            aprint(f"Number of data points             : {x.shape[0]}")
            aprint(f"Number of features per data points: {x.shape[-1]}")

            # Number of features:
            num_of_features = x.shape[-1]

            # How much memory is available:
            max_mem_in_bytes = PerceptronRegressor._get_device_max_mem()

            # We limit ourselves to using only a quarter of memory:
            max_number_of_floats = (max_mem_in_bytes // 4) // 4

            # Max size of batch:
            max_batch_size = max_number_of_floats / num_of_features

            # Batch size taking all this into account:
            batch_size = max(1, min(max_batch_size, x.shape[0] // 256))

            # Heuristic threshold
            batch_size = min(batch_size, (700000 * max_mem_in_bytes) // 12884901888)
            batch_size = int(batch_size)

            aprint(f"Batch size: {batch_size}")
            aprint(f"Predicting. features shape = {x.shape}")

            aprint("NN regressor predicting now...")
            self.model.eval()
            self.model = self.model.to(self.device)

            x_tensor = torch.from_numpy(x.astype(numpy.float32))
            results = []
            with torch.no_grad():
                for i in range(0, len(x_tensor), batch_size):
                    batch = x_tensor[i : i + batch_size].to(self.device)
                    pred = self.model(batch)
                    results.append(pred.cpu().numpy())

            yp = numpy.concatenate(results, axis=0)
            aprint("NN regressor predicting done!")

            # We cast back yp to the correct type and range:
            if self.original_y_dtype is not None:
                yp = yp.astype(self.original_y_dtype)
                yp *= self.original_y_scale

            return yp
