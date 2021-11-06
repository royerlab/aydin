from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso

from aydin.regression.base import RegressorBase
from aydin.util.log.log import lsection, lprint


class LinearRegressor(RegressorBase):
    """Linear Regressor.
    \n\n
    Note: Fast but overall poor performance -- as expected.
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


        kwargs
        """
        super().__init__()
        self.mode = mode
        self.max_num_iterations = max_num_iterations
        self.alpha = alpha
        self.beta = beta

        with lsection("Linear Regressor"):
            lprint(f"mode : {self.mode}")
            lprint(f"alpha: {self.alpha}")

    def _fit(
        self, x_train, y_train, x_valid=None, y_valid=None, regressor_callback=None
    ):
        """Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).
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
    def __init__(self, model):
        self.model = model
        self.loss_history = {'training': [], 'validation': []}

    def _save_internals(self, path: str):
        pass

    def _load_internals(self, path: str):
        pass

    def predict(self, x):

        with lsection("Linear regressor prediction"):

            lprint(f"Number of data points             : {x.shape[0]}")
            lprint(f"Number of features per data points: {x.shape[-1]}")

            with lsection("Linear prediction now"):
                prediction = self.model.predict(x)

            lprint("Linear regressor predicting done!")
            return prediction
