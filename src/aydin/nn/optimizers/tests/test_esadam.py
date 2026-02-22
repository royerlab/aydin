"""Tests for ESAdam optimizer."""

import torch
from torch import nn

from aydin.nn.optimizers.esadam import ESAdam


def _make_simple_model():
    """Create a simple linear model for testing."""
    model = nn.Linear(4, 2)
    return model


def test_esadam_updates_params():
    """Parameters should change after an optimization step."""
    model = _make_simple_model()
    opt = ESAdam(model.parameters(), lr=0.01, start_noise_level=0.001)

    initial_params = [p.clone() for p in model.parameters()]

    x = torch.randn(8, 4)
    y = model(x)
    loss = y.sum()
    loss.backward()
    opt.step()

    for p_init, p_new in zip(initial_params, model.parameters()):
        assert not torch.equal(p_init, p_new)


def test_esadam_step_counter():
    """Step counter should increment with each step."""
    model = _make_simple_model()
    opt = ESAdam(model.parameters(), lr=0.01)
    assert opt.step_counter == 0

    x = torch.randn(8, 4)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    assert opt.step_counter == 1

    opt.zero_grad()
    loss = model(x).sum()
    loss.backward()
    opt.step()
    assert opt.step_counter == 2


def test_esadam_noise_decay():
    """Noise level should decay with step count."""
    model = _make_simple_model()
    opt = ESAdam(model.parameters(), lr=0.01, start_noise_level=1.0)

    # Noise at step 0 = 1.0 / (1 + 0) = 1.0
    # Noise at step 10 = 1.0 / (1 + 10) ~ 0.09
    # The noise decays as 1/(1+step)
    assert opt.start_noise_level / (1 + 0) == 1.0
    assert opt.start_noise_level / (1 + 10) < 0.1


def test_esadam_skips_grad_none():
    """Parameters without gradients should not be modified by noise."""
    model = nn.Linear(4, 2, bias=False)
    # Add a parameter that we don't compute grad for
    extra_param = nn.Parameter(torch.ones(3))
    opt = ESAdam([{'params': model.parameters()}, {'params': [extra_param]}], lr=0.01)

    # Only compute loss through model, extra_param has no grad
    x = torch.randn(8, 4)
    loss = model(x).sum()
    loss.backward()

    extra_before = extra_param.clone()
    opt.step()
    # extra_param should remain unchanged (no grad, so noise is skipped)
    assert torch.equal(extra_before, extra_param)


def test_esadam_state_dict_roundtrip():
    """step_counter survives save/load cycle."""
    model = _make_simple_model()
    opt = ESAdam(model.parameters(), lr=0.01, start_noise_level=0.005)

    # Advance step counter
    for _ in range(5):
        x = torch.randn(4, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    assert opt.step_counter == 5

    state = opt.state_dict()
    assert 'step_counter' in state
    assert state['step_counter'] == 5

    # Create a fresh optimizer and load state
    opt2 = ESAdam(model.parameters(), lr=0.01, start_noise_level=0.005)
    assert opt2.step_counter == 0
    opt2.load_state_dict(state)
    assert opt2.step_counter == 5


def test_esadam_load_state_dict_without_step_counter():
    """Loading a state_dict without step_counter defaults to 0 (backward compat)."""
    model = _make_simple_model()
    opt = ESAdam(model.parameters(), lr=0.01)

    # Build a state_dict without step_counter (simulating old checkpoint)
    state = opt.state_dict()
    state.pop('step_counter', None)

    opt2 = ESAdam(model.parameters(), lr=0.01)
    opt2.step_counter = 99  # set to non-zero to verify it gets reset
    opt2.load_state_dict(state)
    assert opt2.step_counter == 0
