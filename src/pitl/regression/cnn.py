from __future__ import absolute_import, print_function

from lightgbm import LGBMRegressor


class CNNRegressor:
    """
    Regressor that uses the CNN.

      """

    lgbmr: LGBMRegressor

    def __init__(self,
                 num_leaves=63,
                 max_depth=-1,
                 n_estimators=128,
                 learning_rate=0.05,
                 eval_metric='l1',
                 early_stopping_rounds=5
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
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.lgbmr = LGBMRegressor(num_leaves=num_leaves,
                                   max_depth=max_depth,
                                   learning_rate=learning_rate,
                                   n_estimators=n_estimators,
                                   boosting_type='gbdt')

    def fit(self, x_train, y_train, x_test, y_test):
        """
        Fits function y=f(x) goiven training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).

        :param x_train:
        :type x_train:
        :param y_train:
        :type y_train:
        :param x_test:
        :type x_test:
        :param y_test:
        :type y_test:
        """
        self.lgbmr = self.lgbmr.fit(x_train, y_train,
                                    eval_metric=self.eval_metric,
                                    eval_set=[(x_test, y_test)],
                                    early_stopping_rounds=self.early_stopping_rounds)

    def predict(self, x):
        """
        Predicts y given x by applying the learned function f: y=f(x)
        :param x:
        :type x:
        :return:
        :rtype:
        """
        return self.lgbmr.predict(x, num_iteration=self.lgbmr.best_iteration_)
