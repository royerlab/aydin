"""Tests for the PyTorch JINet model forward pass and methods."""

import pytest
import torch

from aydin.nn.models.jinet import JINetModel


@pytest.mark.parametrize(
    "input_array_shape, spacetime_ndim",
    [
        ((1, 1, 64, 64), 2),
        ((1, 1, 32, 32, 32), 3),
    ],
)
def test_forward(input_array_shape, spacetime_ndim):
    """Test that JINet forward pass preserves input shape and dtype."""
    input_array = torch.randn(input_array_shape)
    model = JINetModel(spacetime_ndim=spacetime_ndim)
    result = model(input_array)

    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
    assert torch.isfinite(result).all()
    assert not torch.allclose(result, torch.zeros_like(result))


@pytest.mark.parametrize("spacetime_ndim", [2, 3])
def test_enforce_blind_spot(spacetime_ndim):
    """Test that enforce_blind_spot zeros the center weight."""
    model = JINetModel(spacetime_ndim=spacetime_ndim)
    model.enforce_blind_spot()

    first_conv = model.dilated_conv_functions[0].conv
    center = tuple((k - 1) // 2 for k in first_conv.kernel_size)
    if spacetime_ndim == 2:
        center_weight = first_conv.weight[:, :, center[0], center[1]]
    else:
        center_weight = first_conv.weight[:, :, center[0], center[1], center[2]]

    assert center_weight.abs().sum().item() == 0.0


@pytest.mark.parametrize("spacetime_ndim", [2, 3])
def test_post_optimisation(spacetime_ndim):
    """Test that post_optimisation runs without error."""
    model = JINetModel(spacetime_ndim=spacetime_ndim)
    # Should not raise
    model.post_optimisation()


@pytest.mark.parametrize("spacetime_ndim", [2, 3])
def test_post_optimisation_disabled(spacetime_ndim):
    """Test that post_optimisation is a no-op when disabled."""
    model = JINetModel(
        spacetime_ndim=spacetime_ndim,
        kernel_continuity_regularisation=False,
    )
    weights_before = model.dilated_conv_functions[0].conv.weight.clone()
    model.post_optimisation()
    weights_after = model.dilated_conv_functions[0].conv.weight

    assert torch.equal(weights_before, weights_after)


@pytest.mark.parametrize("spacetime_ndim", [2, 3])
def test_fill_blind_spot(spacetime_ndim):
    """Test that fill_blind_spot restores a non-zero center weight."""
    model = JINetModel(spacetime_ndim=spacetime_ndim)
    model.enforce_blind_spot()

    # Verify center is zero
    first_conv = model.dilated_conv_functions[0].conv
    center = tuple((k - 1) // 2 for k in first_conv.kernel_size)
    if spacetime_ndim == 2:
        assert first_conv.weight[:, :, center[0], center[1]].abs().sum().item() == 0.0
    else:
        assert (
            first_conv.weight[:, :, center[0], center[1], center[2]].abs().sum().item()
            == 0.0
        )

    # Record per-output-channel sums before filling
    weights = first_conv.weight
    out_ch = weights.shape[0]
    sums_before = [weights[oc].sum().item() for oc in range(out_ch)]

    # Fill and verify center is now non-zero
    model.fill_blind_spot()
    if spacetime_ndim == 2:
        center_val = first_conv.weight[:, :, center[0], center[1]].abs().sum().item()
    else:
        center_val = (
            first_conv.weight[:, :, center[0], center[1], center[2]].abs().sum().item()
        )

    assert center_val > 0.0

    # Verify per-channel weight sums are preserved within 1%
    weights_after = first_conv.weight
    for oc in range(out_ch):
        sum_before = sums_before[oc]
        sum_after = weights_after[oc].sum().item()
        if abs(sum_before) > 1e-10:
            rel_error = abs(sum_after - sum_before) / abs(sum_before)
            assert rel_error < 0.01, (
                f"Output channel {oc}: sum changed from {sum_before} to {sum_after} "
                f"(relative error {rel_error:.4f})"
            )


@pytest.mark.parametrize("spacetime_ndim", [2, 3])
def test_dense_layer_channel_dimensions(spacetime_ndim):
    """Test that all kernel_one_conv layers have nb_channels in/out dimensions."""
    model = JINetModel(spacetime_ndim=spacetime_ndim)

    for i, conv in enumerate(model.kernel_one_conv_functions):
        assert (
            conv.in_channels == model.nb_channels
        ), f"kernel_one_conv[{i}] in_channels={conv.in_channels} != nb_channels={model.nb_channels}"
        assert (
            conv.out_channels == model.nb_channels
        ), f"kernel_one_conv[{i}] out_channels={conv.out_channels} != nb_channels={model.nb_channels}"


def test_gradient_flow():
    """Verify gradients flow through the entire JINet."""
    input_array = torch.randn(1, 1, 32, 32, requires_grad=True)
    model = JINetModel(spacetime_ndim=2)
    result = model(input_array)
    loss = result.sum()
    loss.backward()
    assert input_array.grad is not None
    assert input_array.grad.shape == input_array.shape
    assert not torch.allclose(input_array.grad, torch.zeros_like(input_array.grad))
