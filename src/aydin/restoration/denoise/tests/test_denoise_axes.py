"""Tests for batch_axes and channel_axes handling across all denoisers.

Covers the critical gaps identified in test coverage:
- Multi-channel images (channel_axes) for all denoiser backends
- High-dimensional images requiring batch_axes for CNN
- Combined batch + channel axis configurations
- Error messages for invalid dimension configurations
- Functional CNN denoiser train/denoise cycles
"""

import numpy
import pytest
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr

from aydin.io.datasets import add_noise, normalise
from aydin.it.transforms.range import RangeTransform

# Small image size for fast tests
_SIZE = 64
_TRANSFORMS = [{"class": RangeTransform, "kwargs": {}}]


def _crop_camera(size=_SIZE):
    """Return a small normalised camera crop."""
    image = normalise(camera().astype(numpy.float32))
    return image[:size, :size]


def _make_rgb(image_2d):
    """Stack a 2D image into 3 channels along last axis → (H, W, 3)."""
    return numpy.stack([image_2d, image_2d * 0.8, image_2d * 0.6], axis=-1)


def _make_batched(image_2d, n_batch=2):
    """Stack a 2D image into a batch along first axis → (B, H, W)."""
    return numpy.stack([image_2d] * n_batch, axis=0)


# ============================================================================
#  Classic denoiser
# ============================================================================


class TestClassicAxes:
    """Test Classic denoiser with various axis configurations."""

    def test_classic_channel_axes_rgb(self):
        """Classic gaussian should handle RGB images with channel_axes."""
        from aydin.restoration.denoise.classic import Classic

        image = _make_rgb(_crop_camera())
        noisy = add_noise(image)

        classic = Classic(variant='gaussian', it_transforms=_TRANSFORMS)
        classic.train(noisy, channel_axes=[False, False, True])
        denoised = classic.denoise(noisy, channel_axes=[False, False, True])

        assert denoised.shape == noisy.shape

    def test_classic_batch_axes(self):
        """Classic gaussian should handle batched images."""
        from aydin.restoration.denoise.classic import Classic

        image = _make_batched(_crop_camera(), n_batch=2)
        noisy = add_noise(image)

        classic = Classic(variant='gaussian', it_transforms=_TRANSFORMS)
        classic.train(noisy, batch_axes=[True, False, False])
        denoised = classic.denoise(noisy, batch_axes=[True, False, False])

        assert denoised.shape == noisy.shape

    def test_classic_batch_and_channel_axes(self):
        """Classic gaussian with both batch and channel axes (B, H, W, C)."""
        from aydin.restoration.denoise.classic import Classic

        image_2d = _crop_camera()
        rgb = _make_rgb(image_2d)
        batched = numpy.stack([rgb, rgb], axis=0)  # (2, H, W, 3)
        noisy = add_noise(batched)

        classic = Classic(variant='gaussian', it_transforms=_TRANSFORMS)
        classic.train(
            noisy,
            batch_axes=[True, False, False, False],
            channel_axes=[False, False, False, True],
        )
        denoised = classic.denoise(
            noisy,
            batch_axes=[True, False, False, False],
            channel_axes=[False, False, False, True],
        )

        assert denoised.shape == noisy.shape


# ============================================================================
#  Noise2Self FGR denoiser
# ============================================================================


class TestFGRAxes:
    """Test Noise2SelfFGR denoiser with various axis configurations."""

    def test_fgr_channel_axes_rgb(self):
        """FGR linear should handle RGB images with channel_axes."""
        from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR

        image = _make_rgb(_crop_camera())
        noisy = add_noise(image)

        n2s = Noise2SelfFGR(variant='linear', it_transforms=_TRANSFORMS)
        n2s.train(noisy, channel_axes=[False, False, True])
        denoised = n2s.denoise(noisy, channel_axes=[False, False, True])

        assert denoised.shape == noisy.shape

    def test_fgr_batch_axes(self):
        """FGR linear should handle batched images."""
        from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR

        image = _make_batched(_crop_camera(), n_batch=2)
        noisy = add_noise(image)

        n2s = Noise2SelfFGR(variant='linear', it_transforms=_TRANSFORMS)
        n2s.train(noisy, batch_axes=[True, False, False])
        denoised = n2s.denoise(noisy, batch_axes=[True, False, False])

        assert denoised.shape == noisy.shape

    def test_fgr_batch_and_channel_axes(self):
        """FGR linear with both batch and channel axes (B, H, W, C)."""
        from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR

        image_2d = _crop_camera()
        rgb = _make_rgb(image_2d)
        batched = numpy.stack([rgb, rgb], axis=0)  # (2, H, W, 3)
        noisy = add_noise(batched)

        n2s = Noise2SelfFGR(variant='linear', it_transforms=_TRANSFORMS)
        n2s.train(
            noisy,
            batch_axes=[True, False, False, False],
            channel_axes=[False, False, False, True],
        )
        denoised = n2s.denoise(
            noisy,
            batch_axes=[True, False, False, False],
            channel_axes=[False, False, False, True],
        )

        assert denoised.shape == noisy.shape


# ============================================================================
#  Noise2Self CNN denoiser
# ============================================================================


class TestCNNAxes:
    """Test Noise2SelfCNN denoiser with various axis configurations."""

    def test_cnn_train_denoise_2d(self):
        """CNN JINet should train and denoise a 2D image."""
        from aydin.restoration.denoise.noise2selfcnn import Noise2SelfCNN

        image = _crop_camera()
        noisy = add_noise(image)

        n2s = Noise2SelfCNN(variant='jinet', it_transforms=_TRANSFORMS)
        n2s.train(noisy)
        denoised = n2s.denoise(noisy)

        denoised = numpy.clip(denoised, 0, 1)
        assert denoised.shape == image.shape
        assert psnr(image, denoised) > psnr(image, numpy.clip(noisy, 0, 1))

    def test_cnn_channel_axes_rgb(self):
        """CNN JINet should handle RGB images with channel_axes."""
        from aydin.restoration.denoise.noise2selfcnn import Noise2SelfCNN

        image = _make_rgb(_crop_camera())
        noisy = add_noise(image)

        n2s = Noise2SelfCNN(variant='jinet', it_transforms=_TRANSFORMS)
        n2s.train(noisy, channel_axes=[False, False, True])
        denoised = n2s.denoise(noisy, channel_axes=[False, False, True])

        assert denoised.shape == noisy.shape

    def test_cnn_batch_axes(self):
        """CNN JINet should handle batched images with batch_axes."""
        from aydin.restoration.denoise.noise2selfcnn import Noise2SelfCNN

        image = _make_batched(_crop_camera(), n_batch=2)
        noisy = add_noise(image)

        n2s = Noise2SelfCNN(variant='jinet', it_transforms=_TRANSFORMS)
        n2s.train(noisy, batch_axes=[True, False, False])
        denoised = n2s.denoise(noisy, batch_axes=[True, False, False])

        assert denoised.shape == noisy.shape

    def test_cnn_batch_and_channel_axes(self):
        """CNN JINet with both batch and channel axes (B, H, W, C)."""
        from aydin.restoration.denoise.noise2selfcnn import Noise2SelfCNN

        image_2d = _crop_camera()
        rgb = _make_rgb(image_2d)
        batched = numpy.stack([rgb, rgb], axis=0)  # (2, H, W, 3)
        noisy = add_noise(batched)

        n2s = Noise2SelfCNN(variant='jinet', it_transforms=_TRANSFORMS)
        n2s.train(
            noisy,
            batch_axes=[True, False, False, False],
            channel_axes=[False, False, False, True],
        )
        denoised = n2s.denoise(
            noisy,
            batch_axes=[True, False, False, False],
            channel_axes=[False, False, False, True],
        )

        assert denoised.shape == noisy.shape

    def test_cnn_rejects_too_many_spacetime_dims(self):
        """CNN should raise ValueError for >3 spacetime dims with helpful message."""
        from aydin.restoration.denoise.noise2selfcnn import Noise2SelfCNN

        # 4D image with no batch/channel axes → 4 spacetime dims
        image = numpy.random.rand(4, 5, 32, 32).astype(numpy.float32)

        n2s = Noise2SelfCNN(variant='jinet', it_transforms=_TRANSFORMS)
        with pytest.raises(ValueError, match="at most 3D spatial data"):
            n2s.train(image)

    def test_cnn_rejects_4d_without_batch_axes(self):
        """CNN should reject 4D image when no batch axes are specified."""
        from aydin.restoration.denoise.noise2selfcnn import Noise2SelfCNN

        image = numpy.random.rand(3, 5, 32, 32).astype(numpy.float32)

        n2s = Noise2SelfCNN(variant='jinet', it_transforms=_TRANSFORMS)
        with pytest.raises(ValueError, match="batch axes"):
            n2s.train(image)

    def test_cnn_4d_with_batch_axes_succeeds(self):
        """CNN should handle 4D image when leading dim is marked as batch."""
        from aydin.restoration.denoise.noise2selfcnn import Noise2SelfCNN

        image_2d = _crop_camera(size=32)
        # Create (2, D, H, W) volume with batch dim
        vol_3d = numpy.stack([image_2d] * 8, axis=0)  # (8, H, W) → 3D
        batched = numpy.stack([vol_3d, vol_3d], axis=0)  # (2, 8, H, W)
        noisy = add_noise(batched)

        n2s = Noise2SelfCNN(variant='jinet', it_transforms=_TRANSFORMS)
        n2s.train(noisy, batch_axes=[True, False, False, False])
        denoised = n2s.denoise(noisy, batch_axes=[True, False, False, False])

        assert denoised.shape == noisy.shape


# ============================================================================
#  CNN translator-level multi-channel tests
# ============================================================================


class TestCNNTranslatorChannels:
    """Test CNN translator directly with multi-channel inputs."""

    def test_jinet_multichannel_train_translate(self):
        """JINet model should accept multi-channel input via nb_in_channels."""
        from aydin.it.cnn_torch import ImageTranslatorCNNTorch

        image = _crop_camera()
        rgb = _make_rgb(image)  # (H, W, 3)
        noisy = add_noise(rgb)

        it = ImageTranslatorCNNTorch(model="jinet")
        it.train(noisy, noisy, channel_axes=[False, False, True])
        denoised = it.translate(noisy, channel_axes=[False, False, True])

        assert denoised.shape == noisy.shape
        # Verify channels are independently denoised (not all copies of one)
        assert not numpy.allclose(denoised[:, :, 0], denoised[:, :, 2], atol=1e-3)

    def test_jinet_multichannel_model_args(self):
        """_get_model_args should include nb_in_channels from channel dim."""
        from aydin.it.cnn_torch import ImageTranslatorCNNTorch

        it = ImageTranslatorCNNTorch(model="jinet")
        # Simulate shape-normalized image: (B=1, C=3, H=64, W=64)
        fake_input = numpy.zeros((1, 3, 64, 64), dtype=numpy.float32)
        args = it._get_model_args(fake_input)

        assert args['spacetime_ndim'] == 2
        assert args['nb_in_channels'] == 3
        assert args['nb_out_channels'] == 3

    def test_jinet_single_channel_model_args(self):
        """_get_model_args should set nb_in_channels=1 for single-channel."""
        from aydin.it.cnn_torch import ImageTranslatorCNNTorch

        it = ImageTranslatorCNNTorch(model="jinet")
        fake_input = numpy.zeros((1, 1, 64, 64), dtype=numpy.float32)
        args = it._get_model_args(fake_input)

        assert args['spacetime_ndim'] == 2
        assert args['nb_in_channels'] == 1
        assert args['nb_out_channels'] == 1

    def test_dncnn_single_channel_ok(self):
        """DnCNN should work fine with single-channel data."""
        from aydin.it.cnn_torch import ImageTranslatorCNNTorch

        it = ImageTranslatorCNNTorch(model="dncnn")
        fake_input = numpy.zeros((1, 1, 64, 64), dtype=numpy.float32)
        args = it._get_model_args(fake_input)

        assert args['spacetime_ndim'] == 2
        # nb_in_channels should be filtered out (DnCNN doesn't accept it)
        assert 'nb_in_channels' not in args

    def test_dncnn_multichannel_rejected(self):
        """DnCNN should raise ValueError for multi-channel input."""
        from aydin.it.cnn_torch import ImageTranslatorCNNTorch

        it = ImageTranslatorCNNTorch(model="dncnn")
        fake_input = numpy.zeros((1, 3, 64, 64), dtype=numpy.float32)

        with pytest.raises(ValueError, match="does not support multi-channel"):
            it._get_model_args(fake_input)

    def test_unet_multichannel_rejected(self):
        """UNet should raise ValueError for multi-channel input."""
        from aydin.it.cnn_torch import ImageTranslatorCNNTorch

        it = ImageTranslatorCNNTorch(model="unet")
        fake_input = numpy.zeros((1, 3, 64, 64), dtype=numpy.float32)

        with pytest.raises(ValueError, match="does not support multi-channel"):
            it._get_model_args(fake_input)

    def test_jinet_multichannel_save_load(self):
        """Save/load roundtrip should preserve channel count."""
        import os
        import tempfile

        from aydin.it.cnn_torch import ImageTranslatorCNNTorch

        image = _crop_camera(size=32)
        rgb = _make_rgb(image)
        noisy = add_noise(rgb)

        it = ImageTranslatorCNNTorch(model="jinet")
        it.train(noisy, noisy, channel_axes=[False, False, True])
        denoised_before = it.translate(noisy, channel_axes=[False, False, True])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_model')
            os.makedirs(save_path)
            it.save(save_path)

            it_loaded = ImageTranslatorCNNTorch.load(save_path)
            denoised_after = it_loaded.translate(
                noisy, channel_axes=[False, False, True]
            )

        numpy.testing.assert_array_almost_equal(
            denoised_before, denoised_after, decimal=2
        )
