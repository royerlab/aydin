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
from aydin.util.log.log import lsection, lprint


class LGBMRegressor(RegressorBase):
    """
    The LightGBM Regressor uses the gradient boosting library <a
    href="https://github.com/microsoft/LightGBM">LightGBM</a> to perform
    regression from a set of feature vectors and target values. LightGBM is a
    solid library but we do yet support GPU training and inference. Because
    of lack of GPU support LightGBM is slower than CatBoost, sometimes
    LightGBM gives better results than Catbboost, but not often enough to
    justify the loss of speed.
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
        num_leaves
            Number of leaves in the decision trees.
            We recommend values between 128 and 512.
            (advanced)

        max_num_estimators
            Maximum number of estimators (trees). Typical values range from 1024
            to 4096. Use larger values for more difficult datasets. If training
            stops exactly at these values that is a sign you need to increase this
            number. Quality of the results typically increases with the number of
            estimators, but so does computation time too.
            We do not recommend using a value of more than 10000.

        max_bin
            Maximum number of allowed bins. The features are quantised into that
            many bins. Higher values achieve better quantisation of features but
            also leads to longer training and more memory consumption. We do not
            recommend changing this parameter.
            (advanced)

        learning_rate
            Learning rate for the catboost model. The learning rate is determined
            automatically if the value None is given. We recommend values around 0.01.
            (advanced)

        loss
            Type of loss to be used. Van be 'l1' for L1 loss (MAE), and 'l2' for
            L2 loss (RMSE), 'huber' for Huber loss, 'poisson' for Poisson loss,
            and 'quantile' for Auantile loss. We recommend using: 'l1'.
            (advanced)

        patience
            Number of rounds after which training stops if no improvement occurs.
            (advanced)

        verbosity
            Verbosity setting of LightGBM.
            (advanced)

        compute_load
            Allowed load on computational resources in percentage, typically used
            for CPU training when deciding on how many available cores to use.
            (advanced)

        inference_mode : str
            Choses inference mode: can be 'lleaves' for the very fast lleaves
            library (only OSX and Linux), 'lgbm' for the standard lightGBM
            inference engine, and 'auto' (or None) tries the best/fastest
            options first and fallback to lightGBM default inference.
            (advanced)

        compute_training_loss : bool
            Flag to tell LightGBM whether to compute training loss or not
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

        with lsection("LGBM Regressor"):
            lprint(f"learning rate: {self.learning_rate}")
            lprint(f"number of leaves: {self.num_leaves}")
            lprint(f"max bin: {self.max_bin}")
            lprint(f"n_estimators: {self.max_num_estimators}")
            lprint(f"patience: {self.early_stopping_rounds}")
            lprint(f"inference_mode: {self.inference_mode}")

    def _get_params(self, num_samples, dtype=numpy.float32):
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

        lprint(f'objective: {self.num_leaves}')

        # Setting max depth:
        max_depth = max(3, int(int(math.log2(self.num_leaves))) - 1)
        lprint(f'max_depth:  {max_depth}')

        # Setting max bin:
        max_bin = 256 if dtype == numpy.uint8 else self.max_bin
        lprint(f'max_bin:    {max_bin}')

        lprint(f'learning_rate:  {self.learning_rate}')
        lprint(f'num_leaves: {self.num_leaves}')

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

            train_dataset = lightgbm.Dataset(x_train, y_train)
            valid_dataset = (
                lightgbm.Dataset(x_valid, y_valid) if has_valid_dataset else None
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
                    callbacks=[
                        lgbm_callback,
                        self.early_stopping_callback,
                        record_evaluation(evals_result),
                    ]
                    if has_valid_dataset
                    else [lgbm_callback],
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
            return _LGBMModel(model, self.inference_mode, loss_history)


class _LGBMModel:
    def __init__(self, model, inference_mode, loss_history):
        self.model: Booster = model
        self.inference_mode = inference_mode
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

            # we decide here what 'auto' means:
            if self.inference_mode == 'auto':
                if x.shape[0] > 5e6:
                    # Lleaves takes a long time to compile models, so only
                    # interesting for very large inferences!
                    self.inference_mode = 'lleaves'
                else:
                    self.inference_mode = 'lgbm'

            lprint("GBM regressor predicting now...")
            if self.inference_mode == 'lleaves' and find_spec('lleaves'):
                try:
                    return self._predict_lleaves(x)
                except Exception:
                    # printing stack trace
                    # traceback.print_exc()
                    lprint("Failed lleaves-based regression!")

            # This must work!
            return self._predict_lgbm(x)

    def _predict_lleaves(self, x):

        with lsection("Attempting lleaves-based regression."):

            # Creating lleaves model and compiling it:
            with lsection("Model saving and compilation"):
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
        prediction = self.model.predict(x, num_iteration=self.model.best_iteration)
        # LGBM is annoying, it spits out float64s
        prediction = prediction.astype(numpy.float32, copy=False)
        lprint("GBM regressor predicting done!")
        return prediction
