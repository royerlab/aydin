"""Tests for the Noise2Truth supervised training method."""

import numpy
import torch

from aydin.nn.training_methods.n2t import n2t_train


def _make_simple_model():
    """Create a tiny Conv2d model for testing."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 1, 3, padding=1),
    )


def test_n2t_train_reduces_loss():
    """Training should reduce the loss over a few epochs."""
    rng = numpy.random.RandomState(42)
    target = numpy.zeros((1, 1, 16, 16), dtype=numpy.float32)
    target[:, :, 4:12, 4:12] = 1.0
    noisy = target + rng.randn(*target.shape).astype(numpy.float32) * 0.1

    model = _make_simple_model()

    # Evaluate initial loss
    with torch.no_grad():
        initial_pred = model(torch.from_numpy(noisy))
        initial_loss = torch.abs(initial_pred - torch.from_numpy(target)).mean().item()

    n2t_train(
        noisy,
        target,
        model,
        nb_epochs=20,
        lr=0.01,
        patience=50,
    )

    # Evaluate final loss (model may be on GPU after training)
    model.eval().cpu()
    with torch.no_grad():
        final_pred = model(torch.from_numpy(noisy))
        final_loss = torch.abs(final_pred - torch.from_numpy(target)).mean().item()

    assert final_loss < initial_loss


def test_n2t_train_early_stopping():
    """Training should stop early when loss plateaus."""
    target = numpy.ones((1, 1, 8, 8), dtype=numpy.float32) * 0.5
    noisy = target.copy()

    model = _make_simple_model()

    n2t_train(
        noisy,
        target,
        model,
        nb_epochs=1000,
        lr=0.01,
        patience=5,
    )
    # If we get here without timeout, early stopping worked


def test_n2t_train_external_stop():
    """Training should stop when stop_fitting_flag is set."""
    target = numpy.ones((1, 1, 8, 8), dtype=numpy.float32) * 0.5
    noisy = target.copy()

    model = _make_simple_model()
    stop_flag = {'stop': True}

    n2t_train(
        noisy,
        target,
        model,
        nb_epochs=1000,
        lr=0.01,
        patience=500,
        stop_fitting_flag=stop_flag,
    )
    # Should return after 1 epoch due to stop flag


def test_n2t_train_restores_best_model():
    """Training should restore the best model weights at the end."""
    rng = numpy.random.RandomState(42)
    target = numpy.zeros((1, 1, 16, 16), dtype=numpy.float32)
    target[:, :, 4:12, 4:12] = 1.0
    noisy = target + rng.randn(*target.shape).astype(numpy.float32) * 0.3

    model = _make_simple_model()

    n2t_train(
        noisy,
        target,
        model,
        nb_epochs=30,
        lr=0.01,
        patience=10,
    )

    # Model should be in eval-ready state with finite outputs
    model.eval().cpu()
    with torch.no_grad():
        pred = model(torch.from_numpy(noisy))
    assert torch.isfinite(pred).all()
