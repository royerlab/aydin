import numpy

from aydin.regression.lgbm import LGBMRegressor
from aydin.util.log.log import lsection, lprint


class RandomForestRegressor(LGBMRegressor):
    """Random Forrest Regressor (uses the LGBM library)."""

    def __init__(
        self,
        num_leaves: int = 1024,
        max_num_estimators: int = 2048,
        max_bin: int = 512,
        learning_rate: float = 0.0001,
        loss: str = 'l1',
        patience: int = 32,
        verbosity: int = 100,
    ):
        """Constructs a Random Forest regressor

        Parameters
        ----------
        num_leaves
            Number of leaves
        max_num_estimators
            Maximum number of estimators
        max_bin
            Maximum number of allowed bins
        learning_rate
            Learning rate for the LightGBM model
        loss
            Type of loss to be used
        patience
            Number of rounds required for early stopping
        verbosity

        """
        super().__init__(
            num_leaves,
            max_num_estimators,
            max_bin,
            learning_rate,
            loss,
            patience,
            verbosity,
        )

        with lsection("Random Forest Regressor"):
            lprint("with no arguments")  # TODO: fix these logs

    def _get_params(self, num_samples, dtype=numpy.float32):
        params = super()._get_params(num_samples, dtype)
        params["boosting_type"] = "rf"
        params["feature_fraction"] = 0.8
        return params
