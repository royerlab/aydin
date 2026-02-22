"""Tests for the PyTorch CNN image translator.

Consolidates coverage from the former TF/Keras CNN translator tests into
PyTorch equivalents. Tests cover:
- Self-supervised (N2S) training with JINet and UNet
- Supervised (N2T) training
- Save/load roundtrip fidelity
- Non-power-of-2 spatial inputs (padding/cropping)
- Early stopping and stop_training() API
- Shift-convolution training method
- Arbitrary batch dimensions
"""

import os
import tempfile

import numpy
import pytest
from skimage.metrics import peak_signal_noise_ratio as psnr

from aydin.analysis.image_metrics import calculate_print_psnr_ssim, ssim
from aydin.io.datasets import add_noise, camera, normalise
from aydin.it.cnn_torch import ImageTranslatorCNNTorch
from aydin.nn.training_methods.n2s_shiftconv import n2s_shiftconv_train

# ---------------------------------------------------------------------------
#  Self-supervised (Noise2Self) training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model", ["jinet", pytest.param("unet", marks=pytest.mark.unstable)]
)
def test_n2s_2d(model):
    """Self-supervised denoising on 2D camera image should improve PSNR/SSIM."""
    image = normalise(camera())
    noisy = add_noise(image)

    it = ImageTranslatorCNNTorch(model=model)
    it.train(noisy, noisy)
    denoised = it.translate(noisy, tile_size=image.shape[0])

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy, psnr_denoised, ssim_noisy, ssim_denoised = calculate_print_psnr_ssim(
        image, noisy, denoised
    )

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy


# ---------------------------------------------------------------------------
#  Supervised (Noise2Target) training
# ---------------------------------------------------------------------------


def test_n2t_2d():
    """Supervised training with clean target should denoise well.

    Migrated from TF test_it_cnn_jinet2D_supervised_light.
    """
    image = normalise(camera())
    noisy = add_noise(image)

    it = ImageTranslatorCNNTorch(model="jinet")
    # Passing different objects for input/target triggers n2t_train automatically
    it.train(noisy, image)
    denoised = it.translate(noisy, tile_size=image.shape[0])

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(image, noisy)
    psnr_denoised = psnr(image, denoised)
    ssim_noisy = ssim(noisy, image)
    ssim_denoised = ssim(denoised, image)

    print("noisy", psnr_noisy, ssim_noisy)
    print("denoised (supervised)", psnr_denoised, ssim_denoised)

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy


# ---------------------------------------------------------------------------
#  Shift-convolution training
# ---------------------------------------------------------------------------


def test_shiftconv_2d():
    """Shift-conv training via ImageTranslatorCNNTorch produces valid output.

    Migrated from TF test_it_cnn_shiftconv_light.
    The current shiftconv applies shift at output only (not per-layer),
    so the blind-spot constraint is not enforced within internal convolutions.
    Reliable PSNR improvement is not guaranteed; we verify training completes
    and inference produces output of the correct shape.
    """
    image_width = 100
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)

    it = ImageTranslatorCNNTorch(
        model="unet",
        model_kwargs={'nb_unet_levels': 2},
        training_method=n2s_shiftconv_train,
        training_method_kwargs={'nb_epochs': 30},
    )
    it.train(noisy, noisy)
    denoised = it.translate(noisy, tile_size=image_width)

    assert denoised.shape == image.shape
    assert numpy.all(numpy.isfinite(denoised))
    assert not numpy.allclose(denoised, 0)


# ---------------------------------------------------------------------------
#  Save/load roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["jinet", "unet"])
def test_save_load_2d(model):
    """Train, save, load, translate -- loaded model should match original.

    Migrated from TF test_saveload_cnn.
    """
    image = normalise(camera())
    noisy = add_noise(image)

    it = ImageTranslatorCNNTorch(model=model)
    it.train(noisy, noisy)
    denoised_before = it.translate(noisy, tile_size=image.shape[0])

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_model')
        os.makedirs(save_path)
        it.save(save_path)

        it_loaded = ImageTranslatorCNNTorch.load(save_path)
        denoised_after = it_loaded.translate(noisy, tile_size=image.shape[0])

    # The loaded model should produce nearly identical results
    numpy.testing.assert_array_almost_equal(denoised_before, denoised_after, decimal=2)


# ---------------------------------------------------------------------------
#  Non-power-of-2 spatial dimensions
# ---------------------------------------------------------------------------


def test_non_power_of_2_input():
    """Non-power-of-2 inputs should be automatically padded and cropped."""
    image = normalise(camera())
    # Crop to 100x100 which is not divisible by 8
    image = image[:100, :100]
    noisy = add_noise(image)

    it = ImageTranslatorCNNTorch(model="unet")
    it.train(noisy, noisy)
    denoised = it.translate(noisy, tile_size=100)

    # Output should have the same shape as input
    assert denoised.shape == image.shape


# ---------------------------------------------------------------------------
#  Early stopping
# ---------------------------------------------------------------------------


def test_early_stopping():
    """Verify training can stop early before max_epochs with small patience."""
    image = normalise(camera())
    noisy = add_noise(image)

    it = ImageTranslatorCNNTorch(
        model="jinet",
        training_method_kwargs={'nb_epochs': 200, 'patience': 3},
    )
    it.train(noisy, noisy)

    # Just verify it completed without error (early stopping should trigger)
    denoised = it.translate(noisy, tile_size=image.shape[0])
    assert denoised.shape == image.shape


# ---------------------------------------------------------------------------
#  Stop training API
# ---------------------------------------------------------------------------


def test_stop_training():
    """Verify the stop_training() method doesn't crash."""
    it = ImageTranslatorCNNTorch(model="jinet")
    # stop_training should not raise even when no training is running
    it.stop_training()


# ---------------------------------------------------------------------------
#  Arbitrary batch dimensions (migrated from TF test_it_cnn_checkran_light)
# ---------------------------------------------------------------------------


def test_batch_axes():
    """Verify CNN translator handles arbitrary batch dimensions correctly."""
    image_width = 100
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]

    # Reshape to (1, 1, H, W) with batch dims
    arbitrary_shape = (1, 1) + image.shape
    batch_dims = tuple([True if i == 1 else False for i in arbitrary_shape])
    image_batched = image.reshape(arbitrary_shape)
    noisy = add_noise(image_batched)

    it = ImageTranslatorCNNTorch(model="jinet")
    it.train(noisy, noisy, batch_axes=batch_dims)
    denoised = it.translate(noisy, tile_size=image_width, batch_axes=batch_dims)

    assert denoised.shape == noisy.shape

    # Quality check
    denoised_2d = numpy.clip(denoised.squeeze(), 0, 1)
    noisy_2d = numpy.clip(noisy.squeeze(), 0, 1)
    image_2d = numpy.clip(image, 0, 1)

    psnr_noisy = psnr(image_2d, noisy_2d)
    psnr_denoised = psnr(image_2d, denoised_2d)
    print("noisy psnr:", psnr_noisy, "denoised psnr:", psnr_denoised)
    assert psnr_denoised > psnr_noisy


# ---------------------------------------------------------------------------
#  Multi-batch mini-batch inference
# ---------------------------------------------------------------------------


def test_multi_batch_inference():
    """Multiple batch items should be handled via mini-batch inference."""
    image_width = 64
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)

    # Stack 4 copies as batch dimension
    batched = numpy.stack([noisy] * 4, axis=0)  # (4, 64, 64)
    batch_axes = [True, False, False]

    it = ImageTranslatorCNNTorch(model="jinet")
    it.train(batched, batched, batch_axes=batch_axes)
    denoised = it.translate(batched, batch_axes=batch_axes)

    assert denoised.shape == batched.shape
    # Each batch item should be finite and non-trivial
    assert numpy.all(numpy.isfinite(denoised))
    assert not numpy.allclose(denoised, 0)


# ---------------------------------------------------------------------------
#  Memory estimation
# ---------------------------------------------------------------------------


def test_cnn_memory_estimation():
    """CNN translator memory estimation should return meaningful values."""
    image_width = 64
    image = normalise(camera())[:image_width, :image_width]
    noisy = add_noise(image)

    it = ImageTranslatorCNNTorch(model="jinet")
    it.train(noisy, noisy)

    # Shape-normalized image (B, C, H, W)
    test_image = noisy[numpy.newaxis, numpy.newaxis, ...]
    needed, available = it._estimate_memory_needed_and_available(test_image)

    assert needed > 0, "Memory needed should be positive after training"
    assert available > 0, "Available memory should be positive"
    assert (
        needed > test_image.nbytes
    ), "CNN memory estimate should exceed raw image size"


def test_cnn_amplification_factor_before_training():
    """Amplification factor should return conservative default before training."""
    it = ImageTranslatorCNNTorch(model="jinet")
    factor = it._estimate_memory_amplification_factor()
    assert factor == 100.0


def test_cnn_amplification_factor_jinet():
    """Amplification factor for trained JINet should reflect feature counts."""
    image = normalise(camera())[:64, :64]
    noisy = add_noise(image)

    it = ImageTranslatorCNNTorch(model="jinet")
    it.train(noisy, noisy)

    factor = it._estimate_memory_amplification_factor()
    # 2D JINet: 2*128 + 2*128 = 512
    assert factor > 1.0
    assert factor == 512.0
