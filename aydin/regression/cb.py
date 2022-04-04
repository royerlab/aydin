import gc
import math
import multiprocessing
import shutil
from os.path import join
from tempfile import mkdtemp
from typing import Sequence, Optional
import numpy
from catboost import CatBoostRegressor, CatBoostError, Pool

from aydin.regression.base import RegressorBase
from aydin.regression.cb_utils.callbacks import CatBoostStopTrainingCallback
from aydin.util.log.log import lsection, lprint


class CBRegressor(RegressorBase):
    """
    The CatBoost Regressor uses the gradient boosting library <a
    href="https://github.com/catboost">CatBoost</a> to perform regression
    from a set of feature vectors and target values. CatBoost main advantage
    is that it is very fast compared to other gradient boosting libraries --
    in particular when GPU acceleration is available. Compared to other
    libraries (lightGBM, XGBoost) it is much easier to ship the GPU enabled
    version because it just works. It performs comparably and sometimes
    better than other libraries like LightGBM.
    """

    model: CatBoostRegressor

    def __init__(
        self,
        num_leaves: int = 512,
        max_num_estimators: Optional[int] = None,
        min_num_estimators: Optional[int] = None,
        max_bin: int = None,
        learning_rate: Optional[float] = None,
        loss: str = 'l1',
        patience: int = 32,
        compute_load: float = 0.95,
        gpu: bool = True,
        gpu_devices: Sequence[int] = None,
    ):
        """Constructs a CatBoost regressor.

        Parameters
        ----------
        num_leaves : int
            Number of leaves.
            (advanced)
        max_num_estimators : Optional[int]
            Maximum number of estimators.
        min_num_estimators : Optional[int]
            Minimum number of estimators.
            (advanced)
        max_bin : int
            Maximum number of allowed bins
            (advanced)
        learning_rate : Optional[float]
            Learning rate for the catboost model. The learning rate is determined
            automatically if the value None is given.
            (advanced)
        loss : str
            Type of loss to be used
            (advanced)
        patience : int
            Number of rounds required for early stopping
            (advanced)
        compute_load : float
            Allowed load on computational resources in percentage
            (advanced)
        gpu : bool
            True enables GPU acceleration if available
            (advanced)
        gpu_devices : Sequence[int]
            List of GPU device indices to be used by CatBoost
            (advanced)
        """
        super().__init__()

        self.force_verbose_eval = False
        self.stop_training_callback = CatBoostStopTrainingCallback()

        self.num_leaves = num_leaves
        self.max_num_estimators = max_num_estimators
        self.min_num_estimators = min_num_estimators
        if max_bin is None:
            self.max_bin = 254 if gpu else 512
        else:
            self.max_bin = max_bin
        self.learning_rate = learning_rate
        self.metric = loss
        self.early_stopping_rounds = patience
        self.compute_load = compute_load

        self.gpu = gpu
        self.gpu_devices = gpu_devices

        with lsection("CB Regressor"):
            lprint(f"patience: {self.early_stopping_rounds}")
            lprint(f"gpu: {self.gpu}")

    def recommended_max_num_datapoints(self) -> int:
        """Recommended maximum number of datapoints

        Returns
        -------
        int

        """
        return int(40e6 if self.gpu else 1e6)

    def _get_params(
        self, num_samples, num_features, learning_rate, dtype, use_gpu, train_folder
    ):

        min_data_in_leaf = 20 + int(0.01 * (num_samples / self.num_leaves))
        # lprint(f'min_data_in_leaf: {min_data_in_leaf}')

        objective = self.metric
        if objective == 'l1':
            objective = 'MAE'
        elif objective == 'l2':
            objective = 'RMSE'

        max_depth = max(3, int(math.log2(self.num_leaves)) - 1)
        max_depth = min(max_depth, 8) if use_gpu else max_depth

        gpu_ram_type = 'CpuPinnedMemory' if num_samples > 10e6 else 'GpuRam'

        if self.max_num_estimators is None:
            iterations = 4096 if use_gpu else 2048
        else:
            iterations = self.max_num_estimators

        params = {
            "iterations": iterations,
            "task_type": "GPU" if use_gpu else "CPU",
            "devices": 'NULL'
            if self.gpu_devices is None
            else ':'.join(self.gpu_devices),  # uses all available GPUs
            'objective': objective,
            "loss_function": self.metric.upper(),
            "allow_writing_files": True,
            "train_dir": train_folder,
            "max_bin": self.max_bin,
            "rsm": None if use_gpu else 0.8,  # same as GBM
            "thread_count": max(
                1, int(self.compute_load * multiprocessing.cpu_count())
            ),
            "gpu_cat_features_storage": gpu_ram_type,
            'max_depth': max_depth,
            'early_stopping_rounds': self.early_stopping_rounds,
            'bagging_temperature': 1,
            'min_data_in_leaf': min_data_in_leaf,
            'l2_leaf_reg': 30,
            'feature_border_type': 'UniformAndQuantiles',
            # 'verbose_eval' : 10,
            'metric_period': 50 if use_gpu else 1,
            # "num_leaves": self.num_leaves,
            "learning_rate": learning_rate,
        }

        # Note: we could add optional automatic meta-parameter tunning by using cross val:
        # https://effectiveml.com/using-grid-search-to-optimise-catboost-parameters.html

        return params

    def stop_fit(self):
        self.stop_training_callback.continue_training = False

    def _fit(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):

        with lsection("CatBoost regressor fitting:"):

            nb_data_points = y_train.shape[0]
            self.num_features = x_train.shape[-1]
            has_valid_dataset = x_valid is not None and y_valid is not None

            lprint(f"Number of data points: {nb_data_points}")
            if has_valid_dataset:
                lprint(f"Number of validation data points: {y_valid.shape[0]}")
            lprint(f"Number of features per data point: {self.num_features}")

            # Train folder to store training info:
            train_folder = mkdtemp(prefix="catboost_training_")

            self.__epoch_counter = 0

            model = None

            with lsection(
                f"CatBoost regressor fitting now using {f'GPU({self.gpu_devices})' if self.gpu else 'CPU'} "
            ):
                # CatBoost prefers float32 arrays:
                x_train = x_train.astype(numpy.float32, copy=False)
                y_train = y_train.astype(numpy.float32, copy=False)

                xy_train_pool = Pool(data=x_train, label=y_train)

                # Keep this for later:
                x_train_shape = x_train.shape
                y_train_shape = y_train.shape
                x_train_dtype = x_train.dtype

                # Give a chance to reclaim this memory if needed:
                x_train, y_train = None, None

                # CatBoost fails (best_iter == 0 or too small) sometimes to train if learning rate is too high, this loops
                # tries increasingly smaller learning rates until training succeeds (best_iter>min_n_estimators)
                learning_rate = self.learning_rate

                # Default min num of estimators:
                if self.min_num_estimators is None:
                    min_num_estimators = 1024 if self.gpu else 512
                else:
                    min_num_estimators = self.min_num_estimators

                for i in range(10):
                    if not self.stop_training_callback.continue_training:
                        break
                    lprint(
                        f"Trying learning rate of '{learning_rate}' (None -> automatic)"
                    )

                    # The purpose of this try block is to protect against failure to use GPU.
                    try:
                        params = self._get_params(
                            num_samples=nb_data_points,
                            num_features=self.num_features,
                            learning_rate=learning_rate,
                            dtype=x_train_dtype,
                            use_gpu=self.gpu,
                            train_folder=train_folder,
                        )
                        lprint(f"Initialising CatBoost with {params}")
                        model = CatBoostRegressor(**params)

                        lprint(
                            f"Fitting CatBoost model for: X{x_train_shape} -> y{y_train_shape}"
                        )
                        model.fit(
                            X=xy_train_pool,
                            eval_set=(x_valid, y_valid) if has_valid_dataset else None,
                            early_stopping_rounds=self.early_stopping_rounds,
                            use_best_model=has_valid_dataset,
                            # callbacks=[self.stop_training_callback],
                        )
                    except CatBoostError as e:
                        print(e)
                        lprint("GPU training likely failed, switching to CPU.")
                        self.gpu = False
                        # next attempt next...
                        continue

                    # Training succeeds when the best iteration is not the zeroth's iteration.
                    # best_iteration_ might be None if there is no validation data provided...
                    if (
                        model.best_iteration_ is None
                        or model.best_iteration_ > min_num_estimators
                    ):
                        self.learning_rate = learning_rate
                        lprint(
                            f"CatBoost fitting succeeded! new learning rate for regressor: {learning_rate}"
                        )
                        break
                    else:
                        # Reduce learning rate:
                        if learning_rate is None:
                            # If None we were using an automatic value, we set the leraning rate so we can start
                            # with the (relatively high) default value of 0.1
                            learning_rate = 2 * 0.1
                        learning_rate *= 0.5
                        lprint(
                            f"CatBoost fitting failed! best_iteration=={model.best_iteration_} < {self.min_num_estimators} reducing learning rate to: {learning_rate}"
                        )
                        gc.collect()

                lprint("CatBoost fitting done.")

            if has_valid_dataset and model is not None:
                valid_loss = model.get_best_score()['validation'][params['objective']]
                self.last_valid_loss = valid_loss

            loss_history = _read_loss_history(train_folder)
            if (
                'catboost_training_' in train_folder
            ):  # sanity check as we delete a lot of files!
                shutil.rmtree(train_folder, ignore_errors=True)

            gc.collect()
            return _CBModel(model, loss_history)


def _read_loss_history(train_folder):
    training_loss = numpy.genfromtxt(
        join(train_folder, "learn_error.tsv"), delimiter="\t", skip_header=1
    )[:, 1]
    validation_loss = numpy.genfromtxt(
        join(train_folder, "test_error.tsv"), delimiter="\t", skip_header=1
    )[:, 1]
    return {'training': training_loss, 'validation': validation_loss}


class _CBModel:
    def __init__(self, model, loss_history):
        self.model: CatBoostRegressor = model
        self.loss_history = loss_history

    def _save_internals(self, path: str):
        if self.model is not None:
            cb_model_file = join(path, 'catboost_model.txt')
            self.model.save_model(cb_model_file)

    def _load_internals(self, path: str):
        cb_model_file = join(path, 'catboost_model.txt')
        self.model = CatBoostRegressor()
        self.model.load_model(cb_model_file)

    # We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        return state

    def predict(self, x):

        with lsection("CatBoost regressor prediction"):

            lprint(f"Number of data points             : {x.shape[0]}")
            lprint(f"Number of features per data points: {x.shape[-1]}")

            lprint("Converting input to CatBoost's Pool format...")
            # CatBoost prefers float32 arrays:
            x = x.astype(dtype=numpy.float32, copy=False)
            # Create pool object:
            x_pool = Pool(data=x)

            with lsection("CatBoost prediction now"):
                prediction = self.model.predict(
                    x_pool, thread_count=-1, verbose=True
                ).astype(numpy.float32, copy=False)
                # task_type='CPU') # YOUHOUU!

            lprint("CatBoost regressor predicting done!")
            return prediction
