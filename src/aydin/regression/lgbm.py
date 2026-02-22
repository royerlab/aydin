"""LightGBM gradient boosting regressor for Aydin's FGR pipeline.

This module provides :class:`LGBMRegressor`, which wraps the LightGBM library
for gradient-boosted decision-tree regression. Optional accelerated inference
via the ``lleaves`` library is supported on Linux and macOS.
"""

import gc
import math
import multiprocessing
import tempfile
from importlib.util import find_spec
from os.path import join
from typing import Optional

import lightgbm
import numpy
from lightgbm import Booster, record_evaluation

from aydin.regression.base import RegressorBase
from aydin.regression.gbm_utils.callbacks import early_stopping
from aydin.util.log.log import aprint, asection


class LGBMRegressor(RegressorBase):
    """LightGBM gradient-boosted decision-tree regressor.

    Uses the <a href="https://github.com/microsoft/LightGBM">LightGBM</a>
    library to perform regression from a set of feature vectors and target
    values. LightGBM is a solid library but GPU training and inference are
    not yet supported in this wrapper. Because of the lack of GPU support,
    LightGBM is slower than CatBoost in most scenarios. LightGBM sometimes
    gives better results than CatBoost, but not often enough to justify the
    loss of speed.

    Optionally, the `lleaves <https://github.com/siboehm/lleaves>`_ library
    can be used for LLVM-compiled accelerated inference on Linux and macOS.
    <notgui>
    """

    def __init__(
        self,
        num_leaves: Optional[int] = None,
        max_num_estimators: Optional[int] = None,
        max_bin: int = 512,
        learning_rate: Optional[float] = None,
        loss: str = 'l1',
        patience: int = 5,
        verbosity: int = -1,
        compute_load: float = 0.95,
        inference_mode: str = None,
        compute_training_loss: bool = False,
    ):
        """Constructs a LightGBM regressor.

        Parameters
        ----------
        num_leaves : int, optional
            Number of leaves in the decision trees.
            We recommend values between 128 and 512.
            (advanced)
        max_num_estimators : int, optional
            Maximum number of estimators (trees). Typical values range from
            1024 to 4096. Use larger values for more difficult datasets. If
            training stops exactly at these values that is a sign you need
            to increase this number. Quality of the results typically
            increases with the number of estimators, but so does computation
            time too. We do not recommend using a value of more than 10000.
        max_bin : int
            Maximum number of allowed bins. The features are quantised into
            that many bins. Higher values achieve better quantisation of
            features but also leads to longer training and more memory
            consumption. We do not recommend changing this parameter.
            (advanced)
        learning_rate : float, optional
            Learning rate for the LightGBM model. The learning rate is
            determined automatically if the value ``None`` is given. We
            recommend values around 0.01.
            (advanced)
        loss : str
            Type of loss to be used. Can be ``'l1'`` for L1 loss (MAE),
            ``'l2'`` for L2 loss (RMSE), ``'huber'`` for Huber loss,
            ``'poisson'`` for Poisson loss, and ``'quantile'`` for Quantile
            loss. We recommend using ``'l1'``.
            (advanced)
        patience : int
            Number of rounds after which training stops if no improvement
            occurs.
            (advanced)
        verbosity : int
            Verbosity setting of LightGBM.
            (advanced)
        compute_load : float
            Allowed load on computational resources as a fraction (0--1),
            typically used for CPU training when deciding on how many
            available cores to use.
            (advanced)
        inference_mode : str, optional
            Chooses inference mode: ``'lleaves'`` for the very fast lleaves
            library (Linux and macOS only), ``'lgbm'`` for the standard
            LightGBM inference engine, and ``'auto'`` (or ``None``) tries
            the best/fastest options first and falls back to LightGBM
            default inference.
            (advanced)
        compute_training_loss : bool
            Flag to tell LightGBM whether to compute training loss or not.
            (advanced)
        """
        super().__init__()

        self.force_verbose_eval = False

        self.num_leaves = 512 if num_leaves is None else num_leaves
        self.max_num_estimators = (
            int(1e4) if max_num_estimators is None else max_num_estimators
        )
        self.max_bin = max_bin
        self.learning_rate = 0.01 if learning_rate is None else learning_rate
        self.metric = loss
        self.early_stopping_rounds = patience
        self.verbosity = verbosity
        self.compute_load = compute_load
        self.inference_mode = 'auto' if inference_mode is None else inference_mode
        self.compute_training_loss = compute_training_loss  # This can be expensive

        with asection("LGBM Regressor"):
            aprint(f"learning rate: {self.learning_rate}")
            aprint(f"number of leaves: {self.num_leaves}")
            aprint(f"max bin: {self.max_bin}")
            aprint(f"n_estimators: {self.max_num_estimators}")
            aprint(f"patience: {self.early_stopping_rounds}")
            aprint(f"inference_mode: {self.inference_mode}")

    def __repr__(self):
        """Return a concise string representation of the regressor."""
        return (
            f"<{self.__class__.__name__},"
            f" max_num_estimators={self.max_num_estimators},"
            f" lr={self.learning_rate}>"
        )

    def _get_params(self, num_samples, dtype=numpy.float32):
        """Build the LightGBM parameter dictionary for training.

        Parameters
        ----------
        num_samples : int
            Number of training samples.
        dtype : numpy.dtype
            Data type of the training features. Used to adjust ``max_bin``
            for integer types.

        Returns
        -------
        dict
            LightGBM training parameters.
        """
        # min_data_in_leaf = 20 + int(0.01 * (num_samples / self.num_leaves))

        # Preparing objective:
        objective = self.metric
        if objective.lower() == 'l1':
            objective = 'regression_l1'
        elif objective.lower() == 'l2':
            objective = 'regression_l2'
        elif objective.lower() == 'huber':
            objective = 'huber'
        elif objective.lower() == 'poisson':
            objective = 'poisson'
        elif objective.lower() == 'quantile':
            objective = 'quantile'
        else:
            objective = 'regression_l1'

        aprint(f'objective: {objective}')

        # Setting max depth:
        max_depth = max(3, int(int(math.log2(self.num_leaves))) - 1)
        aprint(f'max_depth:  {max_depth}')

        # Setting max bin:
        max_bin = 256 if dtype == numpy.uint8 else self.max_bin
        aprint(f'max_bin:    {max_bin}')

        aprint(f'learning_rate:  {self.learning_rate}')
        aprint(f'num_leaves: {self.num_leaves}')

        params = {
            "device": "cpu",
            "boosting_type": "gbdt",
            'objective': objective,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": max_depth,
            "max_bin": max_bin,
            "subsample_for_bin": 200000,
            "num_threads": max(1, int(self.compute_load * multiprocessing.cpu_count())),
            "metric": self.metric.lower(),
            'verbosity': -1,
            "bagging_freq": 1,
            "bagging_fraction": 0.8,
            "lambda_l1": 0.01,
            "lambda_l2": 0.01,
        }

        return params

    def _fit(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):
        """Fit a single-channel LightGBM model.

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
            Callback invoked at each boosting iteration.

        Returns
        -------
        _LGBMModel
            Fitted LightGBM model wrapper.
        """
        with asection("GBM regressor fitting:"):

            nb_data_points = y_train.shape[0]
            self.num_features = x_train.shape[-1]
            has_valid_dataset = x_valid is not None and y_valid is not None

            aprint(f"Number of data points: {nb_data_points}")
            if has_valid_dataset:
                aprint(f"Number of validation data points: {y_valid.shape[0]}")
            aprint(f"Number of features per data point: {self.num_features}")

            train_dataset = lightgbm.Dataset(x_train, y_train)
            valid_dataset = (
                lightgbm.Dataset(x_valid, y_valid) if has_valid_dataset else None
            )

            self.__epoch_counter = 0

            # We translate the it fgr callback into a lightGBM callback:
            # This avoids propagating annoying 'evaluation_result_list[0][2]'
            # throughout the codebase...
            def lgbm_callback(env):
                """LightGBM callback that forwards metrics to the regressor callback.

                Extracts the validation loss from the LightGBM environment
                object and delegates to ``regressor_callback`` if provided,
                otherwise logs the loss directly.

                Parameters
                ----------
                env : lightgbm.callback.CallbackEnv
                    LightGBM callback environment containing ``iteration``,
                    ``evaluation_result_list``, and ``model``.
                """
                try:
                    val_loss = env.evaluation_result_list[0][2]
                except Exception as e:
                    val_loss = 0
                    aprint("Problem with getting loss from LightGBM 'env' in callback")
                    aprint(str(e))
                if regressor_callback:
                    regressor_callback(env.iteration, val_loss, env.model)
                else:
                    aprint(f"Epoch {self.__epoch_counter}: Validation loss: {val_loss}")
                    self.__epoch_counter += 1

            evals_result = {}

            self.early_stopping_callback = early_stopping(
                self, self.early_stopping_rounds
            )

            with asection("GBM regressor fitting now:"):
                model = lightgbm.train(
                    params=self._get_params(nb_data_points, dtype=x_train.dtype),
                    init_model=None,
                    train_set=train_dataset,
                    valid_sets=(
                        [valid_dataset, train_dataset]
                        if self.compute_training_loss
                        else valid_dataset
                    ),
                    num_boost_round=self.max_num_estimators,
                    callbacks=(
                        [
                            lgbm_callback,
                            self.early_stopping_callback,
                            record_evaluation(evals_result),
                        ]
                        if has_valid_dataset
                        else [lgbm_callback]
                    ),
                )
                aprint("GBM fitting done.")

            del train_dataset
            del valid_dataset

            if has_valid_dataset:
                self.last_valid_loss = evals_result['valid_0'][self.metric][-1]

            if self.compute_training_loss:
                loss_history = {
                    'training': evals_result['training'][self.metric],
                    'validation': evals_result['valid_0'][self.metric],
                }
            else:
                loss_history = {'validation': evals_result['valid_0'][self.metric]}

            gc.collect()
            return _LGBMModel(model, self.inference_mode, loss_history)


class _LGBMModel:
    """Internal wrapper around a fitted LightGBM Booster model.

    Handles serialisation, deserialisation, and prediction (with optional
    ``lleaves`` acceleration) for a single output channel.

    Attributes
    ----------
    model : lightgbm.Booster
        The underlying LightGBM model.
    inference_mode : str
        Inference backend: ``'lgbm'``, ``'lleaves'``, or ``'auto'``.
    loss_history : dict
        Training and/or validation loss arrays.
    """

    def __init__(self, model, inference_mode, loss_history):
        """Initialise the LightGBM model wrapper.

        Parameters
        ----------
        model : lightgbm.Booster
            Trained LightGBM Booster model.
        inference_mode : str
            Inference backend: ``'lgbm'``, ``'lleaves'``, or ``'auto'``.
        loss_history : dict
            Dictionary of training/validation loss arrays.
        """
        self.model: Booster = model
        self.inference_mode = inference_mode
        self.loss_history = loss_history

    def _save_internals(self, path: str):
        """Save the LightGBM model file to the given directory.

        Parameters
        ----------
        path : str
            Directory in which to write ``lgbm_model.txt``.
        """
        if self.model is not None:
            lgbm_model_file = join(path, 'lgbm_model.txt')
            self.model.save_model(lgbm_model_file)

    def _load_internals(self, path: str):
        """Load the LightGBM model file from the given directory.

        Parameters
        ----------
        path : str
            Directory containing ``lgbm_model.txt``.
        """
        lgbm_model_file = join(path, 'lgbm_model.txt')
        self.model = Booster(model_file=lgbm_model_file)

    def __getstate__(self):
        """Return pickling state, excluding the non-serialisable Booster model.

        Returns
        -------
        dict
            Instance state with the ``model`` key removed.
        """
        state = self.__dict__.copy()
        del state['model']
        return state

    def __setstate__(self, state):
        """Restore pickling state with a safe default for the excluded model.

        Parameters
        ----------
        state : dict
            Pickled state (without ``model``).
        """
        self.__dict__.update(state)
        self.model = None

    def predict(self, x):
        """Predict target values for the given feature vectors.

        Uses ``lleaves`` for accelerated inference when the data is large
        enough and the library is available; otherwise falls back to native
        LightGBM prediction.

        Parameters
        ----------
        x : numpy.ndarray
            Feature vectors of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Predicted values as float32 array.
        """
        with asection("GBM regressor prediction:"):

            aprint(f"Number of data points             : {x.shape[0]}")
            aprint(f"Number of features per data points: {x.shape[-1]}")

            # we decide here what 'auto' means:
            inference_mode = self.inference_mode
            if inference_mode == 'auto':
                if x.shape[0] > 5e6:
                    # Lleaves takes a long time to compile models, so only
                    # interesting for very large inferences!
                    inference_mode = 'lleaves'
                else:
                    inference_mode = 'lgbm'

            aprint("GBM regressor predicting now...")
            if inference_mode == 'lleaves' and find_spec('lleaves'):
                try:
                    return self._predict_lleaves(x)
                except Exception:
                    # printing stack trace
                    # traceback.print_exc()
                    aprint("Failed lleaves-based regression!")

            # This must work!
            return self._predict_lgbm(x)

    def _predict_lleaves(self, x):
        """Predict using the lleaves LLVM-compiled inference engine.

        Parameters
        ----------
        x : numpy.ndarray
            Feature vectors.

        Returns
        -------
        numpy.ndarray
            Predicted values.
        """

        with asection("Attempting lleaves-based regression."):

            # Creating lleaves model and compiling it:
            with asection("Model saving and compilation"):
                # Creating temporary file:
                with tempfile.NamedTemporaryFile() as temp_file:

                    # Saving LGBM model:
                    self.model.save_model(
                        temp_file.name, num_iteration=self.model.best_iteration
                    )

                    import lleaves

                    llvm_model = lleaves.Model(model_file=temp_file.name)
                    llvm_model.compile()

            prediction = llvm_model.predict(x)

        return prediction

    def _predict_lgbm(self, x):
        """Predict using the native LightGBM inference engine.

        Parameters
        ----------
        x : numpy.ndarray
            Feature vectors.

        Returns
        -------
        numpy.ndarray
            Predicted values as float32 array.
        """
        prediction = self.model.predict(x, num_iteration=self.model.best_iteration)
        # LGBM is annoying, it spits out float64s
        prediction = prediction.astype(numpy.float32, copy=False)
        aprint("GBM regressor predicting done!")
        return prediction
