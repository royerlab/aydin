from __future__ import absolute_import, print_function

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso
from sklearn.svm import SVR, LinearSVR


class RandomForrestRegressor:
    """
    Random Forrest Regressor.

    Note: definitely ver very slow (much slower than LGBM). Only usefull for reference...
    Also, does not work great for some reason....

    TODO: expose more parameters maybe?

    """

    rf: RandomForestRegressor

    def __init__(self, linear = True):
        """
        Constructs a random forrest regressor.

        """

        self.rf = RandomForestRegressor(max_depth=None,
                                        n_estimators=128)


    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        """
        Fits function y=f(x) goiven training pairs (x_train, y_train).
        Stops when performance stops improving on the validation dataset: (x_valid, y_valid).

        """
        self.rf = self.rf.fit(x_train, y_train)

    def predict(self, x):
        """
        Predicts y given x by applying the learned function f: y=f(x)

        """
        return self.rf.predict(x)
