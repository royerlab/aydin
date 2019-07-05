import math
import multiprocessing

from pitl.regression.gbm import GBMRegressor


class RandomForrestRegressor(GBMRegressor):
    """
    Random Forrest Regressor.


    """

    def __init__(self, num_leaves=63,
                 n_estimators=128,
                 max_bin=512,
                 learning_rate=0.05,
                 metric='l1',
                 early_stopping_rounds=5,
                 verbosity=100):

        super().__init__(num_leaves,
                         n_estimators,
                         max_bin,
                         learning_rate,
                         metric,
                         early_stopping_rounds,
                         verbosity)


    def _get_params(self, num_samples, batch=False):
        params = super()._get_params(num_samples,batch)

        params["boosting_type"] = "rf"
        return params


