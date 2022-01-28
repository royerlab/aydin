from sklearn.svm import SVR, LinearSVR

from aydin.regression.base import RegressorBase
from aydin.util.log.log import lprint, lsection


class SupportVectorRegressor(RegressorBase):
    """Support Vector Regressor.
    \n\n
    Note: Way too slow when non-linear, nearly useless...
    When using linear much faster, but does not perform better
    than linear regression.
    """

    def __init__(self, linear: bool = True):
        """Constructs a linear regressor.

        Parameters
        ----------
        linear : bool
            Flag to choose between a linear or non-linear SVR

        """
        super().__init__()
        self.linear = linear

    def _fit(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):
        """Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).
        """

        if self.linear:
            model = LinearSVR()  # gamma='scale')
        else:
            model = SVR(gamma='scale')

        model.fit(x_train, y_train)

        return _SVRModel(model)


class _SVRModel:
    def __init__(self, model):
        self.model = model
        self.loss_history = {'training': [], 'validation': []}

    def _save_internals(self, path: str):
        pass

    def _load_internals(self, path: str):
        pass

    def predict(self, x):
        with lsection("SVR regressor prediction"):

            lprint(f"Number of data points             : {x.shape[0]}")
            lprint(f"Number of features per data points: {x.shape[-1]}")

            with lsection("SVR prediction now"):
                prediction = self.model.predict(x)

            lprint("SVR regressor predicting done!")
            return prediction
