"""PyTorch feed-forward model architectures for the perceptron regressor.

Provides a feed-forward residual network as a PyTorch ``nn.Module``
for use with :class:`~aydin.regression.perceptron.PerceptronRegressor`.
"""

import torch
from torch import nn


class FeedForwardModel(nn.Module):
    """Feed-forward residual network for regression.

    Architecture: Input → optional GaussianNoise → ``depth`` blocks of
    ``Linear(width) + LeakyReLU`` with additive residual connections
    → ``Linear(1)`` output.

    All intermediate block outputs are summed (residual connections)
    before the final linear layer.

    Parameters
    ----------
    n_features : int
        Number of input features.
    depth : int
        Number of dense blocks.
    noise : float or None
        Standard deviation of Gaussian noise added to the input during
        training. If ``None``, no noise is added.
    weight_decay : float
        L1 regularisation weight (applied as explicit penalty in the
        training loop, not via the optimizer).
    """

    def __init__(self, n_features, depth=16, noise=None, weight_decay=0.0001):
        """Construct a feed-forward residual network.

        Parameters
        ----------
        n_features : int
            Number of input features.
        depth : int
            Number of dense blocks.
        noise : float or None
            Standard deviation of Gaussian noise added to input during
            training. If ``None``, no noise is added.
        weight_decay : float
            L1 regularisation weight applied as explicit penalty.
        """
        super().__init__()
        self.n_features = n_features
        self.depth = depth
        self.noise_std = noise
        self.weight_decay = weight_decay

        width = n_features

        # Optional Gaussian noise layer
        if noise is not None and noise > 0:
            self.noise_layer = _GaussianNoise(noise)
        else:
            self.noise_layer = None

        # Dense blocks: each is Linear + LeakyReLU
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(width, width, bias=False),
                    nn.LeakyReLU(negative_slope=0.01),
                )
            )

        # Final output layer
        self.fc_last = nn.Linear(width, 1)

    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, n_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch_size, 1)``.
        """
        if self.noise_layer is not None:
            x = self.noise_layer(x)

        outputs = [x]
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        # Sum all intermediate outputs (residual connections)
        x = torch.stack(outputs, dim=0).sum(dim=0)

        x = self.fc_last(x)
        return x

    def l1_penalty(self):
        """Compute L1 regularisation penalty over all linear layer weights.

        Returns the mean absolute weight value scaled by ``weight_decay``,
        providing a scale-invariant regularisation term.

        Returns
        -------
        torch.Tensor
            Scalar L1 penalty value.
        """
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                penalty = penalty + module.weight.abs().sum()
                count += module.weight.numel()
                if module.bias is not None:
                    penalty = penalty + module.bias.abs().sum()
                    count += module.bias.numel()
        if count > 0:
            penalty = penalty / count
        return self.weight_decay * penalty


class _GaussianNoise(nn.Module):
    """Add Gaussian noise during training only.

    Parameters
    ----------
    std : float
        Standard deviation of the Gaussian noise to add.
    """

    def __init__(self, std):
        """Construct the Gaussian noise layer.

        Parameters
        ----------
        std : float
            Standard deviation of the noise.
        """
        super().__init__()
        self.std = std

    def forward(self, x):
        """Apply Gaussian noise to the input during training.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Input tensor with added noise (training) or unchanged (eval).
        """
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x
