"""Support vector regressor for Aydin's FGR pipeline.

This module provides :class:`SupportVectorRegressor`, which wraps scikit-learn's
SVR and LinearSVR. This regressor is generally too slow for practical use and
is included primarily for benchmarking purposes.
"""

from sklearn.svm import SVR, LinearSVR

from aydin.regression.base import RegressorBase
from aydin.util.log.log import lprint, lsection


class SupportVectorRegressor(RegressorBase):
    """
    The Support Vector Regressor is too slow and does not in our experience
    perform better than random forests or gradient boosting.
    """

    def __init__(self, linear: bool = True):
        """Construct a support vector regressor.

        Parameters
        ----------
        linear : bool
            Flag to choose between a linear or non-linear SVR
            (advanced)

        """
        super().__init__()
        self.linear = linear

    def __repr__(self):
        return f"<{self.__class__.__name__}, linear={self.linear}>"

    def _fit(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):
        """Fit a single-channel support vector regression model.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training feature vectors of shape ``(n_samples, n_features)``.
        y_train : numpy.ndarray
            Training target values of shape ``(n_samples,)``.
        x_valid : numpy.ndarray, optional
            Validation feature vectors (unused by SVR).
        y_valid : numpy.ndarray, optional
            Validation target values (unused by SVR).
        regressor_callback : callable, optional
            Callback (unused by SVR).

        Returns
        -------
        _SVRModel
            Fitted SVR model wrapper.
        """

        if self.linear:
            model = LinearSVR()  # gamma='scale')
        else:
            model = SVR(gamma='scale')

        model.fit(x_train, y_train)

        return _SVRModel(model)


class _SVRModel:
    """Internal wrapper around a fitted scikit-learn SVR model.

    Attributes
    ----------
    model : object
        The underlying scikit-learn SVR or LinearSVR estimator.
    loss_history : dict
        Empty loss history (SVR does not track iterative loss).
    """

    def __init__(self, model):
        self.model = model
        self.loss_history = {'training': [], 'validation': []}

    def _save_internals(self, path: str):
        """Save model internals (no-op for scikit-learn models)."""
        pass

    def _load_internals(self, path: str):
        """Load model internals (no-op for scikit-learn models)."""
        pass

    def predict(self, x):
        """Predict target values for the given feature vectors.

        Parameters
        ----------
        x : numpy.ndarray
            Feature vectors of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Predicted values.
        """
        with lsection("SVR regressor prediction"):

            lprint(f"Number of data points             : {x.shape[0]}")
            lprint(f"Number of features per data points: {x.shape[-1]}")

            with lsection("SVR prediction now"):
                prediction = self.model.predict(x)

            lprint("SVR regressor predicting done!")
            return prediction
