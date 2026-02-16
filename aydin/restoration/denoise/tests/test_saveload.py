"""Tests for saving and loading denoiser models.

Verifies that each denoiser type (Classic, Noise2SelfFGR, Noise2SelfCNN)
can be trained, saved to a zip archive, loaded back, and still produce
equivalent denoising results.
"""

import time
from os.path import join

import numpy
import pytest
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr

from aydin import Classic
from aydin.analysis.image_metrics import ssim
from aydin.io.datasets import add_noise, normalise
from aydin.io.folders import get_temp_folder
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.restoration.denoise.noise2selfcnn import Noise2SelfCNN
from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR

transforms = [
    {"class": RangeTransform, "kwargs": {}},
    {"class": PaddingTransform, "kwargs": {}},
]


@pytest.mark.parametrize(
    "denoiser, min_psnr, min_ssim",
    [
        (Classic(variant="gaussian", it_transforms=transforms), 19, 0.40),
        (Noise2SelfFGR(variant="cb", it_transforms=transforms), 20, 0.55),
        (Noise2SelfCNN(it_transforms=transforms), 13, 0.12),
    ],
)
def test_saveload(denoiser, min_psnr, min_ssim):
    """Test that a denoiser can be saved, reloaded, and still denoise well.

    Verifies that: (1) the loaded model produces results matching the
    original, (2) denoising improves quality over noisy input, and
    (3) quality meets minimum thresholds for each denoiser type.

    Parameters
    ----------
    denoiser : DenoiseRestorationBase
        The denoiser instance to test (Classic, FGR, or CNN).
    min_psnr : float
        Minimum acceptable PSNR for the denoised image.
    min_ssim : float
        Minimum acceptable SSIM for the denoised image.
    """
    image = normalise(camera().astype(numpy.float32))
    noisy = add_noise(image)

    denoiser.train(noisy)
    denoised_before = denoiser.denoise(noisy)
    denoised_before = denoised_before.clip(0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised_before, image)
    ssim_denoised = ssim(denoised_before, image)
    print("denoised (before save)", psnr_denoised, ssim_denoised)

    temp_file = join(get_temp_folder(), "test_restoration_saveload" + str(time.time()))
    print(temp_file)

    denoiser.save(temp_file)
    loaded_denoiser = denoiser.__class__()

    del denoiser

    loaded_denoiser.load(temp_file + ".zip")

    denoised_after = loaded_denoiser.denoise(noisy)
    denoised_after = denoised_after.clip(0, 1)

    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(image, denoised_after)
    ssim_denoised = ssim(image, denoised_after)
    print("denoised (after load)", psnr_denoised, ssim_denoised)

    # Primary check: loaded model produces same results as original
    numpy.testing.assert_array_almost_equal(denoised_before, denoised_after, decimal=2)

    # Denoising should improve upon noisy input
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy

    # Quality should meet minimum thresholds for the denoiser type
    assert psnr_denoised > min_psnr and ssim_denoised > min_ssim
