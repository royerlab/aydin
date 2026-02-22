"""Tests for the ImageDenoiserClassic image translator."""

# flake8: noqa
import time

import numpy
import pytest
from skimage.metrics import peak_signal_noise_ratio as psnr

from aydin.analysis.image_metrics import ssim
from aydin.io.datasets import add_noise, cropped_newyork, normalise
from aydin.it.classic import ImageDenoiserClassic


def test_it_classic_gaussian():
    """Test classic image translator with Gaussian denoising method."""
    do_it_classic(method_name="gaussian", min_psnr=15.5, min_ssim=0.30)


def test_it_classic_butterworth():
    """Test classic image translator with Butterworth denoising method."""
    do_it_classic(method_name="butterworth", min_psnr=14.5, min_ssim=0.30)


@pytest.mark.heavy
def test_it_classic_nlm():
    """Test classic image translator with Non-Local Means denoising method."""
    do_it_classic(method_name="nlm", min_psnr=15, min_ssim=0.25)


@pytest.mark.heavy
def test_it_classic_tv():
    """Test classic image translator with Total Variation denoising method."""
    do_it_classic(method_name="tv", min_psnr=15, min_ssim=0.25)


def do_it_classic(method_name, min_psnr=22, min_ssim=0.75):
    """
    Test for self-supervised denoising using camera image with synthetic noise
    """

    image = normalise(cropped_newyork().astype(numpy.float32))
    noisy = add_noise(image)

    it = ImageDenoiserClassic(method=method_name)

    start = time.time()
    it.train(noisy, noisy)
    stop = time.time()
    print(f"####### Training: elapsed time:  {stop - start} sec")

    start = time.time()
    denoised = it.translate(noisy)
    stop = time.time()
    print(f"####### Inference: elapsed time:  {stop - start} sec")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy

    # if the line below fails, then the parameters of the image the regressor have been broken.
    # do not change the number below, but instead, fix the problem -- most likely a parameter.

    assert psnr_denoised > min_psnr and ssim_denoised > min_ssim
