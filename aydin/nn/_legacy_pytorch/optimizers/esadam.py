"""ESAdam optimizer - Adam with exploratory noise injection.

Implements a modified Adam optimizer that adds decaying random noise
to parameters during optimization, encouraging exploration of the
loss landscape.
"""

import torch
from torch.optim import Adam


class ESAdam(Adam):
    """Adam optimizer with exploratory stochastic noise injection.

    Extends the standard Adam optimizer by adding decaying random noise
    to the parameters after each optimization step. The noise level
    decreases over time as ``start_noise_level / (1 + step_counter)``,
    encouraging exploration in early training and convergence later.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize.
    start_noise_level : float
        Initial noise amplitude (decays with training steps).
    **kwargs
        Additional keyword arguments passed to ``torch.optim.Adam``.
    """

    def __init__(self, params, start_noise_level=0.001, **kwargs):
        super().__init__(params, **kwargs)

        self.start_noise_level = start_noise_level
        self.step_counter = 0

    def step(self, closure=None):
        """Perform a single optimization step with noise injection.

        Runs the standard Adam update and then adds decaying random
        noise to all non-sparse parameters.

        Parameters
        ----------
        closure : callable or None
            A closure that re-evaluates the model and returns the loss.

        Returns
        -------
        loss
            The loss value (from the parent Adam step).
        """
        loss = super().step(closure)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad: torch.Tensor = p.grad.data
                if grad.is_sparse:
                    continue

                step_size = group['lr']

                p.data += (
                    step_size
                    * (self.start_noise_level / (1 + self.step_counter))
                    * (torch.rand_like(p.data) - 0.5)
                )

        self.step_counter += 1

        return loss
