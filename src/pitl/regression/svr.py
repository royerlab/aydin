from __future__ import absolute_import, print_function

from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso
from sklearn.svm import SVR, LinearSVR


class SupportVectorRegressor:
    """
    Support Vector Regressor.

    Note: Way too slow when non-linear, nearly useless...
    When using linear much faster, but does not performa better than straight linear regression...


    """

    svr: SVR

    def __init__(self, linear = True):
        """
        Constructs a linear regressor.

        """
        if linear:
            self.svr = LinearSVR() #gamma='scale')
        else:
            self.svr = SVR(gamma='scale')


    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        """
        Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).

        """
        self.svr = self.svr.fit(x_train, y_train)

    def predict(self, x):
        """
        Predicts y given x by applying the learned function f: y=f(x)
        """
        return self.svr.predict(x)
