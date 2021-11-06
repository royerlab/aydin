import torch
from torch.optim import Adam


class ESAdam(Adam):
    r"""Implements a modifified version of the Adam algorithm that adds noise to the
    the .

    """

    def __init__(self, params, start_noise_level=0.001, **kwargs):
        super().__init__(params, **kwargs)

        self.start_noise_level = start_noise_level
        self.step_counter = 0

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
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
