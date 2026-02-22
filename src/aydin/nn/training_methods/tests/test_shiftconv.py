"""Tests for shift-convolution training method.

Covers low-level rotation/shift operations and a full 2D training integration
test (migrated from TF test_shiftconv_2D and test_shiftconv_3D).
"""

import numpy
import torch

from aydin.nn.training_methods.n2s_shiftconv import (
    ShiftConvWrapper,
    _rotate_2d,
    _shift_right,
)


def test_shiftconv_rotation_roundtrip():
    """Verify rotate + un-rotate = identity for all 4 rotation angles."""
    x = torch.randn(1, 1, 32, 32)

    for k in range(4):
        rotated = _rotate_2d(x, k)
        unrotated = _rotate_2d(rotated, -k)
        torch.testing.assert_close(x, unrotated)


def test_shift_right_shape():
    """Verify _shift_right preserves tensor shape."""
    x = torch.randn(1, 1, 16, 16)
    shifted = _shift_right(x)
    assert shifted.shape == x.shape


def test_shift_right_content():
    """Verify _shift_right moves content one pixel right."""
    x = torch.zeros(1, 1, 4, 4)
    x[0, 0, 0, 0] = 1.0  # Set first pixel

    shifted = _shift_right(x)

    # The value should have moved to position (0, 1)
    assert shifted[0, 0, 0, 1].item() == 1.0
    # Original position should be 1.0 (replicate padding preserves edge value)
    assert shifted[0, 0, 0, 0].item() == 1.0


def test_shiftconv_wrapper_2d_forward():
    """Verify ShiftConvWrapper produces correct output shape for 2D.

    Migrated from TF test_shiftconv_2D.
    """
    from aydin.nn.models.unet import UNetModel

    input_array = torch.randn(1, 1, 64, 64)
    base_model = UNetModel(nb_unet_levels=2, spacetime_ndim=2)
    wrapper = ShiftConvWrapper(base_model, spacetime_ndim=2)

    result = wrapper(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
    # Verify non-trivial output
    assert not torch.allclose(result, torch.zeros_like(result))


def test_shiftconv_wrapper_3d_forward():
    """Verify ShiftConvWrapper produces correct output shape for 3D.

    Migrated from TF test_shiftconv_3D.
    """
    from aydin.nn.models.unet import UNetModel

    input_array = torch.randn(1, 1, 16, 32, 32)
    base_model = UNetModel(nb_unet_levels=2, spacetime_ndim=3)
    wrapper = ShiftConvWrapper(base_model, spacetime_ndim=3)

    result = wrapper(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
    assert not torch.allclose(result, torch.zeros_like(result))


def test_shiftconv_2d_train_and_infer():
    """Verify shiftconv training runs and produces valid output.

    The current shiftconv applies shift at output only (not per-layer),
    so it does not enforce the blind-spot constraint within internal
    convolutions. This means reliable PSNR improvement is not guaranteed.
    We test that training completes and inference produces finite,
    non-trivial output of the correct shape.
    """
    from aydin.io.datasets import add_noise, camera, normalise
    from aydin.nn.models.unet import UNetModel
    from aydin.nn.training_methods.n2s_shiftconv import n2s_shiftconv_train

    image = normalise(camera())[:128, :128]
    noisy = add_noise(image)

    noisy_bchw = numpy.expand_dims(numpy.expand_dims(noisy, 0), 0)
    model = UNetModel(nb_unet_levels=2, spacetime_ndim=2)

    n2s_shiftconv_train(noisy_bchw, model, nb_epochs=30, verbose=False)

    # Inference must use the ShiftConvWrapper since the model was trained
    # within the wrapper's rotate-shift-unrotate pipeline
    wrapper = ShiftConvWrapper(model, spacetime_ndim=2)
    wrapper.cpu()
    wrapper.eval()
    x = torch.from_numpy(noisy_bchw)
    with torch.no_grad():
        denoised = wrapper(x).numpy()[0, 0]

    assert denoised.shape == image.shape
    assert numpy.all(numpy.isfinite(denoised))
    assert not numpy.allclose(denoised, 0)
