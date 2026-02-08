"""Linear regression methods for Aydin's FGR pipeline.

This module provides :class:`LinearRegressor`, which wraps scikit-learn's
LinearRegression, HuberRegressor, and Lasso into a unified interface.
"""

from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression

from aydin.regression.base import RegressorBase
from aydin.util.log.log import aprint, asection


class LinearRegressor(RegressorBase):
    """
    The Linear Regressor is the simplest of all regressors, and in general
    performs poorly. However, it is also very fast and can be advantageous in
    some 'simple' situations.
    """

    def __init__(
        self,
        mode: str = 'huber',
        max_num_iterations: int = 512,
        alpha: float = 1,
        beta: float = 0.0001,
        **kwargs,
    ):
        """Constructs a linear regressor

        Parameters
        ----------
        mode : str
            Regression mode, supported options: 'lasso', 'huber', and 'linear'

        max_num_iterations: int
            Maximum number of iterations.

        alpha: float
            Regularisation weight for L1 sparsity term for Lasso regression
            (mode='lasso').

        beta: float
            Regularisation weight for Huber regression (mode='huber').


        **kwargs
            Additional keyword arguments (unused, accepted for interface
            compatibility).
        """
        super().__init__()
        self.mode = mode
        self.max_num_iterations = max_num_iterations
        self.alpha = alpha
        self.beta = beta

        with asection("Linear Regressor"):
            aprint(f"mode : {self.mode}")
            aprint(f"alpha: {self.alpha}")

    def __repr__(self):
        return f"<{self.__class__.__name__}, mode={self.mode}, max_num_iterations={self.max_num_iterations}>"

    def _fit(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):
        """Fit a single-channel linear model.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training feature vectors of shape ``(n_samples, n_features)``.
        y_train : numpy.ndarray
            Training target values of shape ``(n_samples,)``.
        x_valid : numpy.ndarray, optional
            Validation feature vectors (unused by linear models).
        y_valid : numpy.ndarray, optional
            Validation target values (unused by linear models).
        regressor_callback : callable, optional
            Callback (unused by linear models).

        Returns
        -------
        _LinearModel
            Fitted linear model wrapper.

        Raises
        ------
        Exception
            If ``self.mode`` is not one of ``'lasso'``, ``'huber'``, or
            ``'linear'``.
        """
        if self.mode == 'lasso':
            model = Lasso(alpha=self.alpha, max_iter=self.max_num_iterations)
        elif self.mode == 'huber':
            model = HuberRegressor(max_iter=self.max_num_iterations, alpha=self.beta)
        elif self.mode == 'linear':
            model = LinearRegression(n_jobs=-1)
        else:
            raise Exception(f"Unknown mode: {self.mode}!")

        model.fit(x_train, y_train)

        return _LinearModel(model)


class _LinearModel:
    """Internal wrapper around a fitted scikit-learn linear model.

    Attributes
    ----------
    model : object
        The underlying scikit-learn estimator.
    loss_history : dict
        Empty loss history (linear models do not track iterative loss).
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
        with asection("Linear regressor prediction"):

            aprint(f"Number of data points             : {x.shape[0]}")
            aprint(f"Number of features per data points: {x.shape[-1]}")

            with asection("Linear prediction now"):
                prediction = self.model.predict(x)

            aprint("Linear regressor predicting done!")
            return prediction
