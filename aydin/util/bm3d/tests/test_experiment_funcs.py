"""Tests for BM3D experiment utility functions."""

import numpy as np
import pytest

from aydin.util.bm3d.experiment_funcs import (
    get_cropped_psnr,
    get_experiment_kernel,
    get_experiment_noise,
    get_psnr,
)


class TestGetPsnr:
    """Tests for the get_psnr function."""

    def test_identical_signals_give_inf_psnr(self):
        """Identical arrays should produce infinite PSNR."""
        signal = np.random.RandomState(0).rand(32, 32).astype(np.float64)
        psnr = get_psnr(signal, signal)
        assert psnr == np.inf

    def test_known_noise_level(self):
        """PSNR should match expected value for a known MSE."""
        ref = np.ones((64, 64), dtype=np.float64)
        # Add noise with known MSE = 0.01 => PSNR = 10*log10(1/0.01) = 20 dB
        rng = np.random.RandomState(42)
        noise = rng.normal(0, 0.1, size=(64, 64))
        est = ref + noise
        psnr = get_psnr(est, ref)
        # With 64x64 samples, the empirical MSE should be close to 0.01
        assert 18.0 < psnr < 22.0

    def test_psnr_is_symmetric(self):
        """PSNR(a, b) should equal PSNR(b, a)."""
        rng = np.random.RandomState(7)
        a = rng.rand(16, 16)
        b = rng.rand(16, 16)
        assert np.isclose(get_psnr(a, b), get_psnr(b, a))


class TestGetCroppedPsnr:
    """Tests for the get_cropped_psnr function."""

    def test_cropped_psnr_identical(self):
        """Identical signals should still give inf PSNR after cropping."""
        signal = np.random.RandomState(1).rand(32, 32).astype(np.float64)
        psnr = get_cropped_psnr(signal, signal, crop=(4, 4))
        assert psnr == np.inf

    def test_cropped_psnr_differs_from_full(self):
        """Cropped PSNR may differ from full PSNR when border noise differs."""
        rng = np.random.RandomState(99)
        ref = np.ones((32, 32), dtype=np.float64)
        est = ref.copy()
        # Add heavy noise only on the border
        est[:3, :] += rng.normal(0, 1.0, size=(3, 32))
        est[-3:, :] += rng.normal(0, 1.0, size=(3, 32))
        est[:, :3] += rng.normal(0, 1.0, size=(32, 3))
        est[:, -3:] += rng.normal(0, 1.0, size=(32, 3))

        psnr_full = get_psnr(est, ref)
        psnr_cropped = get_cropped_psnr(est, ref, crop=(4, 4))
        # Cropping away the noisy border should give higher PSNR
        assert psnr_cropped > psnr_full


class TestGetExperimentKernel:
    """Tests for the get_experiment_kernel function."""

    def test_white_noise_kernel(self):
        """White noise kernels ('gw', 'g0') should be a single element [[1]]."""
        for noise_type in ('gw', 'g0'):
            kernel = get_experiment_kernel(noise_type, noise_var=1.0)
            # After normalisation the kernel should be [[1.0]]
            assert kernel.shape == (1, 1)
            np.testing.assert_allclose(kernel, [[1.0]], atol=1e-10)

    def test_kernel_variance_scaling(self):
        """Kernel L2 norm squared should equal the noise variance."""
        for noise_type in ('gw', 'g1', 'g2', 'g3'):
            noise_var = 0.25
            kernel = get_experiment_kernel(noise_type, noise_var)
            l2_norm_sq = np.sum(kernel**2)
            np.testing.assert_allclose(l2_norm_sq, noise_var, rtol=1e-5)

    def test_g1_kernel_is_horizontal_line(self):
        """g1 kernel should be a 1D horizontal pattern (one row)."""
        kernel = get_experiment_kernel('g1', noise_var=1.0)
        # g1 is a horizontal line kernel: shape should be (1, 31)
        assert kernel.shape[0] == 1
        assert kernel.shape[1] == 31

    def test_invalid_noise_type_raises(self):
        """Invalid noise types should raise ValueError."""
        with pytest.raises(ValueError, match="Noise type must be one of"):
            get_experiment_kernel('invalid', noise_var=1.0)

    def test_g4_kernel_uses_image_size(self):
        """g4 (pink noise) kernel should respect the sz parameter."""
        sz = (64, 64)
        kernel = get_experiment_kernel('g4', noise_var=1.0, sz=sz)
        # Kernel shape should match or be derived from the provided sz
        assert kernel.shape[0] == sz[0]
        assert kernel.shape[1] == sz[1]


class TestGetExperimentNoise:
    """Tests for the get_experiment_noise function."""

    def test_noise_shape_matches_input(self):
        """Output noise should have the expected spatial shape."""
        sz = (64, 64)
        noise, psd, kernel = get_experiment_noise(
            'gw', noise_var=0.5, realization=0, sz=sz
        )
        # For 2D, get_experiment_noise uses atleast_3d, output may have trailing dim
        assert noise.shape[0] == sz[0]
        assert noise.shape[1] == sz[1]

    def test_noise_is_reproducible(self):
        """Same realization seed should produce identical noise."""
        sz = (32, 32)
        noise1, _, _ = get_experiment_noise('gw', noise_var=1.0, realization=42, sz=sz)
        noise2, _, _ = get_experiment_noise('gw', noise_var=1.0, realization=42, sz=sz)
        np.testing.assert_array_equal(noise1, noise2)

    def test_psd_shape(self):
        """PSD should have spatial dimensions matching sz."""
        sz = (48, 48)
        _, psd, _ = get_experiment_noise('g1', noise_var=0.5, realization=0, sz=sz)
        assert psd.shape[0] == sz[0]
        assert psd.shape[1] == sz[1]
