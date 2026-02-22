"""Tests for center pixel smoothing on JINet models."""

from aydin.nn.models.jinet import JINetModel
from aydin.nn.utils.center_smoothing import apply_center_smoothing


def test_smoothing_modifies_center_weight():
    """Verify that center pixel weight changes from 0 to non-zero after smoothing."""
    model = JINetModel(spacetime_ndim=2)

    # Find first DilatedConv and zero its center pixel
    from aydin.nn.layers.dilated_conv import DilatedConv

    first_dilated = None
    for module in model.modules():
        if isinstance(module, DilatedConv):
            first_dilated = module
            break

    assert first_dilated is not None

    weight = first_dilated.conv.weight.data
    kh, kw = weight.shape[2], weight.shape[3]
    center_h, center_w = kh // 2, kw // 2

    # Zero the center pixel (simulating blind-spot training)
    weight[:, :, center_h, center_w] = 0.0
    assert weight[:, :, center_h, center_w].abs().sum().item() == 0.0

    # Apply smoothing
    result = apply_center_smoothing(model, spacetime_ndim=2)
    assert result is True

    # Center should now be non-zero
    assert weight[:, :, center_h, center_w].abs().sum().item() > 0.0


def test_smoothing_preserves_weight_sum():
    """Verify that total weight sum is preserved within tolerance."""
    model = JINetModel(spacetime_ndim=2)

    from aydin.nn.layers.dilated_conv import DilatedConv

    first_dilated = None
    for module in model.modules():
        if isinstance(module, DilatedConv):
            first_dilated = module
            break

    weight = first_dilated.conv.weight.data
    original_sum = weight.sum().item()

    apply_center_smoothing(model, spacetime_ndim=2)

    new_sum = weight.sum().item()

    # Sum should be preserved within tolerance
    assert abs(original_sum - new_sum) / max(abs(original_sum), 1e-10) < 0.01


def test_smoothing_3d():
    """Verify 3D smoothing works without errors."""
    model = JINetModel(spacetime_ndim=3)

    result = apply_center_smoothing(model, spacetime_ndim=3)
    assert result is True


def test_smoothing_preserves_per_channel_weight_sum():
    """Per-channel weight sum should be preserved within 1% after smoothing."""
    model = JINetModel(spacetime_ndim=2)

    from aydin.nn.layers.dilated_conv import DilatedConv

    first_dilated = None
    for module in model.modules():
        if isinstance(module, DilatedConv):
            first_dilated = module
            break

    assert first_dilated is not None

    weight = first_dilated.conv.weight.data
    out_ch = weight.shape[0]

    # Zero center pixel (simulate blind-spot training)
    kh, kw = weight.shape[2], weight.shape[3]
    center_h, center_w = kh // 2, kw // 2
    weight[:, :, center_h, center_w] = 0.0

    # Record per-channel sums before smoothing
    per_channel_sums_before = [weight[oc].sum().item() for oc in range(out_ch)]

    apply_center_smoothing(model, spacetime_ndim=2)

    # Check per-channel sums are preserved within 1%
    for oc in range(out_ch):
        before = per_channel_sums_before[oc]
        after = weight[oc].sum().item()
        if abs(before) > 1e-10:
            assert (
                abs(before - after) / abs(before) < 0.01
            ), f"Channel {oc}: sum changed from {before} to {after}"
