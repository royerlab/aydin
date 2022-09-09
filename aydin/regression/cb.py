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
        num_leaves: int = None,
        max_num_estimators: Optional[int] = None,
        min_num_estimators: Optional[int] = None,
        max_bin: int = None,
        learning_rate: Optional[float] = None,
        loss: str = 'l1',
        patience: int = 32,
        compute_load: float = 0.95,
        gpu: bool = True,
        gpu_use_pinned_ram: Optional[bool] = None,
        gpu_devices: Optional[Sequence[int]] = None,
    ):
        """Constructs a CatBoost regressor.

        Parameters
        ----------
        num_leaves : int
            Number of leaves in the decision trees.
            We recommend values between 128 and 512.
            (advanced)

        max_num_estimators : Optional[int]
            Maximum number of estimators (trees). Typical values range from 1024
            to 4096. Use larger values for more difficult datasets. If training
            stops exactly at these values that is a sign you need to increase this
            number. Quality of the results typically increases with the number of
            estimators, but so does computation time too.
            We do not recommend using a value of more than 10000.

        min_num_estimators : Optional[int]
            Minimum number of estimators. Training restarts with a lower learning
            rate if the number of estimators is too low as defined by this threshold.
            Regressor that have too few estimators typically lead to poor results.
            (advanced)

        max_bin : int
            Maximum number of allowed bins. The features are quantised into that
            many bins. Higher values achieve better quantisation of features but
            also leads to longer training and more memory consumption. We do not
            recommend changing this parameter.
            When using GPU training the number of bins must be equal or below 254.
            (advanced)

        learning_rate : Optional[float]
            Learning rate for the catboost model. The learning rate is determined
            automatically if the value None is given. We recommend values around 0.01.
            (advanced)

        loss : str
            Type of loss to be used. Van be 'l1' for L1 loss (MAE), and 'l2' for
            L2 loss (RMSE), 'Lq:q=1.5' with q>=1 real number as power coefficient (here q=1.5),
            'Poisson' for Poisson loss, 'Huber:delta=0.1' for Huber loss with delta=0.1,
            'Expectile:alpha=0.5' for expectile loss with alpha parameter set to 0.5,
            or 'expectile' as a shortcut for 'Expectile:alpha=0.5'.
            We recommend using: 'l1', 'l2', and 'Poisson'.
            (advanced)

        patience : int
            Number of rounds after which training stops if no improvement occurs.
            (advanced)

        compute_load : float
            Allowed load on computational resources in percentage, typically used
            for CPU training when deciding on how many available cores to use.
            (advanced)

        gpu : bool
            True enables GPU acceleration if available. Falls back to CPU if it
            fails for any reason.
            (advanced)

        gpu_use_pinned_ram : Optional[bool]
            True forces the usage of CPU pinned memory byte GPU which can be a
            bit slower but also can accommodate larger dataset. By default the
            usage, or not, of CPU pinned memory is determined automatically
            based on size of data and GPU VRAM size. You can override this
            automatic default.
            (advanced)

        gpu_devices : Optional[Sequence[int]]
            List of GPU device indices to be used by CatBoost. For example,
            to use GPUs of index 0 and 1, set to '0:1'. For a range of devices
            set to '0-3' for example for all devices 0,1,2,3. It is recommended
            to only use together similar or ideally identical GPU devices.
            (advanced)


        """
        super().__init__()

        self.force_verbose_eval = False
        self.stop_training_callback = CatBoostStopTrainingCallback()

        # Default value for number of leaves:
        self.num_leaves = 512 if num_leaves is None else num_leaves

        # Default max number of estimators:
        if max_num_estimators is None:
            self.max_num_estimators = 4096 if gpu else 2048
        else:
            self.max_num_estimators = max_num_estimators

        # Default min number of estimators:
        if min_num_estimators is None:
            self.min_num_estimators = 1024 if gpu else 512
        else:
            self.min_num_estimators = min_num_estimators

        # Ensure min is below or equal to max:
        self.max_num_estimators = max(self.min_num_estimators, self.max_num_estimators)
        self.min_num_estimators = min(self.min_num_estimators, self.max_num_estimators)

        # max iterations should not be above 15k in any case:
        self.max_num_estimators = min(self.max_num_estimators, 15000)

        # max bin defaults:
        if max_bin is None:
            self.max_bin = 254 if gpu else 512
        else:
            self.max_bin = max_bin

        # other parameters:
        self.learning_rate = learning_rate
        self.metric = loss
        self.early_stopping_rounds = patience
        self.compute_load = compute_load

        self.gpu = gpu
        self.gpu_use_pinned_ram = gpu_use_pinned_ram
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

    def _get_params(self, num_samples, learning_rate, use_gpu, train_folder):

        # Setting min data in leaf:
        min_data_in_leaf = 20 + int(0.01 * (num_samples / self.num_leaves))
        lprint(f'min_data_in_leaf: {min_data_in_leaf}')

        # Normalise losses/metrics/objectives:
        objective: str = self.metric
        if objective.lower() == 'l1':
            objective = 'MAE'
        elif objective.lower() == 'l2':
            objective = 'RMSE'
        elif objective.lower() == 'poisson':
            objective = 'Poisson'
        elif objective.lower() == 'expectile':
            objective = 'Expectile:alpha=0.5'
        else:
            objective = 'l1'
        lprint(f'objective: {objective}')

        # We pick a max depth:
        max_depth = max(3, int(math.log2(self.num_leaves)) - 1)
        max_depth = min(max_depth, 8) if use_gpu else max_depth
        lprint(f'max_depth: {max_depth}')

        # If the dataset is really big we want to switch to pinned memeory:
        if self.gpu_use_pinned_ram is None:
            gpu_ram_type = 'CpuPinnedMemory' if num_samples > 10e6 else 'GpuRam'
        else:
            gpu_ram_type = 'CpuPinnedMemory' if self.gpu_use_pinned_ram else 'GpuRam'
        lprint(f'gpu_ram_type: {gpu_ram_type}')

        # Setting max number of iterations:
        iterations = self.max_num_estimators
        lprint(f'max_num_estimators: {iterations}')

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

        # Note: we could add optional automatic meta-parameter tuning by using cross val:
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
                # x_train_dtype = x_train.dtype

                # Give a chance to reclaim this memory if needed:
                x_train, y_train = None, None

                # CatBoost fails (best_iter == 0 or too small) sometimes to train
                # if learning rate is too high, this loops tries increasingly smaller
                # learning rates until training succeeds (best_iter>min_n_estimators)
                learning_rate = self.learning_rate

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
                            learning_rate=learning_rate,
                            use_gpu=self.gpu,
                            train_folder=train_folder,
                        )
                        lprint(f"Initialising CatBoost with {params}")
                        model = CatBoostRegressor(**params)

                        # Logging callback:
                        class MetricsCheckerCallback:
                            def after_iteration(self, info):
                                iteration = info.iteration
                                metrics = info.metrics
                                lprint(f"Iteration: {iteration} metrics: {metrics}")
                                return True

                        # Callbacks:
                        callbacks = None if self.gpu else [MetricsCheckerCallback()]  #

                        # When to be silent? when we actually can printout the logs.
                        silent = not self.gpu

                        lprint(
                            f"Fitting CatBoost model for: X{x_train_shape} -> y{y_train_shape}"
                        )
                        model.fit(
                            X=xy_train_pool,
                            eval_set=(x_valid, y_valid) if has_valid_dataset else None,
                            early_stopping_rounds=self.early_stopping_rounds,
                            use_best_model=has_valid_dataset,
                            callbacks=callbacks,
                            silent=silent,
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
                        or model.best_iteration_ > self.min_num_estimators
                    ):
                        self.learning_rate = learning_rate
                        lprint(
                            f"CatBoost fitting succeeded! new learning rate for regressor: {learning_rate}"
                        )
                        break
                    else:
                        # Reduce learning rate:
                        if learning_rate is None:
                            # If None we were using an automatic value, we set the learning rate so we can start
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

            def _predict(task_type):
                return self.model.predict(
                    x_pool,
                    thread_count=-1 if task_type == 'CPU' else 1,
                    verbose=True,
                    task_type=task_type,
                ).astype(numpy.float32, copy=False)

            with lsection("CatBoost prediction now"):
                prediction = _predict('CPU')

                # Unfortunately this does not work yet, please keep code for when it does...
                # try:
                #     lprint("Trying GPU inference...")
                #     prediction = _predict('GPU')
                #     lprint("Success!")
                # except:
                #     lprint("GPU inference failed, trying CPU inference instead...")
                #     prediction = _predict('CPU')

            lprint("CatBoost regressor predicting done!")
            return prediction
