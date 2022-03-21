import gc
import math
import multiprocessing
from os.path import join

import lightgbm
import numpy
from lightgbm import Booster

from aydin.regression.base import RegressorBase
from aydin.regression.gbm_utils.callbacks import early_stopping
from aydin.util.log.log import lsection, lprint


class LGBMRegressor(RegressorBase):
    """LightGBM Regressor."""

    def __init__(
        self,
        num_leaves: int = 127,
        max_num_estimators: int = int(1e4),
        max_bin: int = 512,
        learning_rate: float = 0.01,
        loss: str = 'l1',
        patience: int = 5,
        verbosity: int = -1,
        compute_load: float = 0.95,
        gpu_prediction: bool = False,
        compute_training_loss: bool = False,
    ):
        """Constructs a LightGBM regressor.

        Parameters
        ----------
        num_leaves
            Number of leaves
        max_num_estimators
            Maximum number of estimators
        max_bin
            Maximum number of allowed bins
        learning_rate
            Learning rate for the LightGBM model
        loss
            Type of loss to be used
        patience
            Number of rounds required for early stopping
        verbosity
        compute_load
            Allowed load on computational resources in percentage
        gpu_prediction : bool
        compute_training_loss : bool
            Flag to tell LightGBM whether to compute training loss or not

        """
        super().__init__()

        self.force_verbose_eval = False

        self.num_leaves = num_leaves
        self.max_num_estimators = max_num_estimators
        self.max_bin = max_bin
        self.learning_rate = learning_rate
        self.metric = loss
        self.early_stopping_rounds = patience
        self.verbosity = verbosity
        self.compute_load = compute_load
        self.gpu_prediction = gpu_prediction
        self.compute_training_loss = compute_training_loss  # This can be expensive

        self.opencl_predictor = None

        with lsection("LGBM Regressor"):
            lprint(f"learning rate: {self.learning_rate}")
            lprint(f"number of leaves: {self.num_leaves}")
            lprint(f"max bin: {self.max_bin}")
            lprint(f"n_estimators: {self.max_num_estimators}")
            lprint(f"patience: {self.early_stopping_rounds}")

    def _get_params(self, num_samples, dtype=numpy.float32):
        # min_data_in_leaf = 20 + int(0.01 * (num_samples / self.num_leaves))
        max_depth = max(3, int(int(math.log2(self.num_leaves))) - 1)
        max_bin = 256 if dtype == numpy.uint8 else self.max_bin

        lprint(f'learning_rate:  {self.learning_rate}')
        lprint(f'max_depth:  {max_depth}')
        lprint(f'num_leaves: {self.num_leaves}')
        lprint(f'max_bin:    {max_bin}')

        objective = self.metric
        if objective == 'l1':
            objective = 'regression_l1'
        elif objective == 'l2':
            objective = 'regression_l2'

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

        with lsection("GBM regressor fitting:"):

            nb_data_points = y_train.shape[0]
            self.num_features = x_train.shape[-1]
            has_valid_dataset = x_valid is not None and y_valid is not None

            lprint(f"Number of data points: {nb_data_points}")
            if has_valid_dataset:
                lprint(f"Number of validation data points: {y_valid.shape[0]}")
            lprint(f"Number of features per data point: {self.num_features}")

            train_dataset = lightgbm.Dataset(x_train, y_train, silent=True)
            valid_dataset = (
                lightgbm.Dataset(x_valid, y_valid, silent=True)
                if has_valid_dataset
                else None
            )

            self.__epoch_counter = 0

            # We translate the it fgr callback into a lightGBM callback:
            # This avoids propagating annoying 'evaluation_result_list[0][2]'
            # throughout the codebase...
            def lgbm_callback(env):
                try:
                    val_loss = env.evaluation_result_list[0][2]
                except Exception as e:
                    val_loss = 0
                    lprint("Problem with getting loss from LightGBM 'env' in callback")
                    print(str(e))
                if regressor_callback:
                    regressor_callback(env.iteration, val_loss, env.model)
                else:
                    lprint(f"Epoch {self.__epoch_counter}: Validation loss: {val_loss}")
                    self.__epoch_counter += 1

            evals_result = {}

            verbose_eval = (lgbm_callback is None) or (self.force_verbose_eval)

            self.early_stopping_callback = early_stopping(
                self, self.early_stopping_rounds
            )

            with lsection("GBM regressor fitting now:"):
                model = lightgbm.train(
                    params=self._get_params(nb_data_points, dtype=x_train.dtype),
                    init_model=None,
                    train_set=train_dataset,
                    valid_sets=[valid_dataset, train_dataset]
                    if self.compute_training_loss
                    else valid_dataset,
                    early_stopping_rounds=None if has_valid_dataset else None,
                    num_boost_round=self.max_num_estimators,
                    callbacks=[lgbm_callback, self.early_stopping_callback]
                    if has_valid_dataset
                    else [lgbm_callback],
                    verbose_eval=verbose_eval,
                    evals_result=evals_result,
                )
                lprint("GBM fitting done.")

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
            return _LGBMModel(model, self.gpu_prediction, loss_history)


class _LGBMModel:
    def __init__(self, model, gpu_prediction, loss_history):
        self.model = model
        self.gpu_prediction = gpu_prediction
        self.loss_history = loss_history

    def _save_internals(self, path: str):
        if self.model is not None:
            lgbm_model_file = join(path, 'lgbm_model.txt')
            self.model.save_model(lgbm_model_file)

    def _load_internals(self, path: str):
        lgbm_model_file = join(path, 'lgbm_model.txt')
        self.model = Booster(model_file=lgbm_model_file)

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        return state

    def predict(self, x):
        with lsection("GBM regressor prediction:"):

            lprint(f"Number of data points             : {x.shape[0]}")
            lprint(f"Number of features per data points: {x.shape[-1]}")

            lprint("GBM regressor predicting now...")
            if self.gpu_prediction:
                try:
                    lprint("Attempting OpenCL-based regression.")
                    from aydin.regression.gbm_utils.opencl_prediction import (
                        GBMOpenCLPrediction,
                    )

                    if self.opencl_predictor is None:
                        self.opencl_predictor = GBMOpenCLPrediction()

                    prediction = self.opencl_predictor.predict(
                        self.model, x, num_iteration=self.model.best_iteration
                    )

                    # We clear the OpenCL ressources:
                    del self.opencl_predictor
                    self.opencl_predictor = None

                    return prediction
                except Exception:
                    lprint(
                        "Failed OpenCL-based regression, doing CPU based prediction."
                    )

            prediction = self.model.predict(x, num_iteration=self.model.best_iteration)
            # LGBM is annoying, it spits out float64s
            prediction = prediction.astype(numpy.float32, copy=False)
            lprint("GBM regressor predicting done!")
            return prediction
