import numpy
from skimage.metrics import structural_similarity

from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.demo.demo_2D_wavelet import demo_wavelet
from aydin.it.classic_denoisers.test.util_test_nd import check_nd
from aydin.it.classic_denoisers.wavelet import denoise_wavelet


def test_wavelet():
    assert demo_wavelet(cropped_newyork(), display=False) >= 0.45


def test_wavelet_nd():
    check_nd(denoise_wavelet)


def test_wavelet_uses_aydin_implementation():
    """Regression test for GitHub issue #116.

    The wavelet test previously imported scikit-image's denoise_wavelet
    instead of aydin's. Verify that our denoise_wavelet is the aydin
    wrapper (which accepts the same API but lives in aydin's module).
    """
    assert denoise_wavelet.__module__ == 'aydin.it.classic_denoisers.wavelet'


def test_wavelet_denoises_noisy_image():
    """Verify wavelet denoiser actually reduces noise on a synthetic image."""
    numpy.random.seed(42)
    clean = numpy.random.rand(64, 64).astype(numpy.float32) * 0.5 + 0.25
    # Smooth the clean image so it has structure
    from scipy.ndimage import gaussian_filter

    clean = gaussian_filter(clean, sigma=2).astype(numpy.float32)
    noisy = clean + numpy.random.normal(0, 0.1, clean.shape).astype(numpy.float32)

    denoised = denoise_wavelet(noisy)

    ssim_noisy = structural_similarity(clean, noisy, data_range=1.0)
    ssim_denoised = structural_similarity(clean, denoised, data_range=1.0)
    # Denoised image should have better SSIM than the noisy input
    assert ssim_denoised > ssim_noisy, (
        f"Wavelet denoiser did not improve SSIM: noisy={ssim_noisy:.3f}, "
        f"denoised={ssim_denoised:.3f}"
    )
