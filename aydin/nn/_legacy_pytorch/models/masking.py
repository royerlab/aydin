"""Random pixel masking wrapper for self-supervised training.

Wraps a model with a random binary mask applied to the input,
implementing a simple blind-spot strategy.
"""

import torch
import torch.nn as nn


class Masking(nn.Module):
    """Random pixel masking wrapper for self-supervised training.

    Wraps a model module and applies a random binary mask to the
    input before passing it through the wrapped model.

    Parameters
    ----------
    module : torch.nn.Module
        The model to wrap with masking.
    """

    def __init__(self, module):
        super().__init__()

        self.module: nn.Module = module
        self.density = 0.1
        self.mask = None

    def forward(self, x):
        """Apply random masking and forward through the wrapped model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the wrapped model on the masked input.
        """
        self.mask = torch.rand_like(x) < self.density
        x = (~self.mask) * x
        x = self.module(x)
        return x

    def get_mask(self):
        """Return the most recently generated mask.

        Returns
        -------
        torch.Tensor
            Boolean mask tensor from the last forward pass.
        """
        return self.mask

    def trainable_parameters(self):
        """Return trainable parameters of the wrapped module.

        Returns
        -------
        iterator
            Iterator over trainable parameters.
        """
        return self.module.trainable_parameters()
