from __future__ import absolute_import, print_function

from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso


class LinearRegressor:
    """
    Linear Regressor.

    Note: Fast but overall poor performance

    """

    linear: LinearRegression

    def __init__(self,
                 mode = 'Lasso'
                 ):
        """
        Constructs a linear regressor.

        """
        if mode=='lasso':
            self.linear = Lasso(alpha=0.1)
        elif mode=='huber':
            self.linear = HuberRegressor()
        elif mode=='linear':
            self.linear = LinearRegression()

    def fit(self, x_train, y_train, x_test=None, y_test=None):
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
        self.linear = self.linear.fit(x_train, y_train)

    def predict(self, x):
        """
        Predicts y given x by applying the learned function f: y=f(x)
        :param x:
        :type x:
        :return:
        :rtype:
        """
        return self.linear.predict(x)
