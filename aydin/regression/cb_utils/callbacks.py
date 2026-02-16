"""CatBoost training callbacks.

Provides a stop-training callback that can be used to gracefully interrupt
CatBoost training from an external thread.
"""

from aydin.util.log.log import aprint


class CatBoostStopTrainingCallback:
    """CatBoost callback that supports externally triggered early stopping.

    Set :attr:`continue_training` to ``False`` from another thread to
    stop the current CatBoost training loop at the end of the current
    iteration.

    Attributes
    ----------
    continue_training : bool
        Flag controlling whether training should continue. Set to
        ``False`` to request a stop.
    """

    def __init__(self):
        """Initialise the callback with training enabled."""
        self.continue_training = True

    def after_iteration(self, info):
        """Called by CatBoost after each boosting iteration.

        Logs current training and test MAE metrics and returns whether
        training should continue.

        Parameters
        ----------
        info : object
            CatBoost callback info object containing ``iteration`` and
            ``metrics`` attributes.

        Returns
        -------
        bool
            ``True`` to continue training, ``False`` to stop.
        """
        aprint(
            f"{info.iteration}: learn: {info.metrics['learn']['MAE'][-1]}  test: {info.metrics['test']['MAE'][-1]}"
        )
        return self.continue_training
