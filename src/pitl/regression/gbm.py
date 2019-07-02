from __future__ import absolute_import, print_function

import gc
import math
import multiprocessing
from typing import List, Union

import lightgbm
from lightgbm import Booster

from pitl.regression.regressor_base import RegressorBase


class GBMRegressor(RegressorBase):
    """
    Regressor that uses the LightGBM library.

    TODO:   (i)   use directly the native lgbm API instead of the skilearn facade.
    TODO:   (ii)  Expose key tuning parameters described here: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    TODO:   (iii) implement increnental learning using technique described here: https://towardsdatascience.com/how-to-learn-from-bigdata-files-on-low-memory-incremental-learning-d377282d38ff
    TODO:   (iv)  for that purpose (point iii) we will need to dump the x_train and y_train to disk and load it incrementally for learning...
    TODO:   (v)   try the GPU acceleration: https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html
    """

    lgbmr: Union[Booster, List[Booster]]

    def __init__(self,
                 num_leaves=63,
                 n_estimators=128,
                 max_bin=512,
                 learning_rate=0.05,
                 eval_metric='l1',
                 early_stopping_rounds=5,
                 verbosity=100
                 ):
        """
        Constructs a LightGBM regressor.

        :param num_leaves:
        :type num_leaves:
        :param n_estimators:
        :type n_estimators:
        :param learning_rate:
        :type learning_rate:
        :param eval_metric:
        :type eval_metric:
        :param early_stopping_rounds:
        :type early_stopping_rounds:
        """

        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.max_bin = max_bin
        self.learning_rate = learning_rate
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.verbosity = verbosity
        self.lgbmr = None

        # LGBMRegressor(num_leaves=num_leaves,
        #                            max_depth=max_depth,
        #                            learning_rate=learning_rate,
        #                            n_estimators=n_estimators,
        #                            boosting_type='gbdt')

    def reset(self):
        del self.lgbmr
        self.lgbmr = []
        gc.collect()


    def _get_params(self, num_samples, batch=False):
        min_data_in_leaf = 20 + int(0.01 * (num_samples / self.num_leaves))
        # print(f'min_data_in_leaf: {min_data_in_leaf}')
        return {'objective': 'regression_l2',
                "boosting_type": "gbdt",
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "max_depth": max(3, int(math.log2(self.num_leaves)) - 1),
                "max_bin": self.max_bin,
                #"min_data_in_leaf": min_data_in_leaf,
                "subsample_for_bin": 200000,
                "num_threads": multiprocessing.cpu_count() // 2,
                "metric": ["l2", "l1"],
                'verbosity': -1  #self.verbosity
                }

    def fit_batch(self, x_train, y_train, x_valid=None, y_valid=None):
        self._fit(x_train, y_train, x_valid, y_valid, is_batch=True)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        self._fit(x_train, y_train, x_valid, y_valid, is_batch=False)

    def _fit(self, x_train, y_train, x_valid=None, y_valid=None, is_batch=False):

        num_samples = y_train.shape[0]
        has_valid_dataset = x_valid is not None and y_valid is not None

        train_dataset = lightgbm.Dataset(x_train, y_train)
        valid_dataset = lightgbm.Dataset(x_valid, y_valid) if has_valid_dataset else None

        model = lightgbm.train(params=self._get_params(num_samples, batch=is_batch),
                               init_model=None,  # self.lgbmr if is_batch else None, <-- not working...
                                    train_set=train_dataset,
                                    valid_sets=valid_dataset,
                               early_stopping_rounds=self.early_stopping_rounds if has_valid_dataset else None,
                                    num_boost_round=self.n_estimators,
                               # keep_training_booster= is_batch, <-- not working...
                                    )

        if is_batch:
            if isinstance(self.lgbmr, (list,)) or self.lgbmr is None:
                self.lgbmr = []
            self.lgbmr.append(model)
        else:
            self.lgbmr = model

        del train_dataset
        del valid_dataset

        gc.collect()

    def predict(self, x):
        """
        Predicts y given x by applying the learned function f: y=f(x)
        :param x:
        :type x:
        :return:
        :rtype:
        """
        if isinstance(self.lgbmr, (list,)):
            yp = None
            counter = 0
            for model in self.lgbmr:
                yp_batch = model.predict(x, num_iteration=model.best_iteration)

                if yp is None:
                    yp = yp_batch
                else:
                    yp += yp_batch

                counter = counter + 1

            yp /= counter

            return yp


        else:
            return self.lgbmr.predict(x, num_iteration=self.lgbmr.best_iteration)
