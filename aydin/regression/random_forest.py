"""Random forest regressor for Aydin's FGR pipeline.

This module provides :class:`RandomForestRegressor`, which uses LightGBM's
random forest boosting mode for fast, decent-quality regression.
"""

import numpy

from aydin.regression.lgbm import LGBMRegressor
from aydin.util.log.log import aprint, asection


class RandomForestRegressor(LGBMRegressor):
    """
    The Random Forest Regressor uses random forest regression as
    implemented in the <a href="https://github.com/microsoft/LightGBM
    ">LightGBM</a> library. This regressor is very fast and has decent
    performance, offering an attractive trade-off between speed and quality
    that is advantageous for 'easy' datasets.
    """

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
            (advanced)
        max_num_estimators
            Maximum number of estimators
        max_bin
            Maximum number of allowed bins
            (advanced)
        learning_rate
            Learning rate for the LightGBM random forest model
            (advanced)
        loss
            Type of loss to be used
            (advanced)
        patience
            Number of rounds required for early stopping
            (advanced)
        verbosity
            Verbosity setting of LightGBM.
            (advanced)

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

        with asection("Random Forest Regressor"):
            aprint("with no arguments")  # TODO: fix these logs

    def __repr__(self):
        return f"<{self.__class__.__name__}, max_num_estimators={self.max_num_estimators}, lr={self.learning_rate}>"

    def _get_params(self, num_samples, dtype=numpy.float32):
        """Build LightGBM parameters with random forest boosting type.

        Extends the parent :meth:`LGBMRegressor._get_params` by switching
        the boosting type to ``"rf"`` and adding a feature fraction setting.

        Parameters
        ----------
        num_samples : int
            Number of training samples.
        dtype : numpy.dtype
            Data type of the training features.

        Returns
        -------
        dict
            LightGBM training parameters configured for random forest mode.
        """
        params = super()._get_params(num_samples, dtype)
        params["boosting_type"] = "rf"
        params["feature_fraction"] = 0.8
        return params
