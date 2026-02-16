"""Tests for image quality metrics (PSNR, mutual information, joint information)."""

import pytest

from aydin.analysis.image_metrics import (
    joint_information,
    mutual_information,
    spectral_mutual_information,
    spectral_psnr,
)
from aydin.io.datasets import add_noise, camera, normalise


def test_spectral_psnr():
    """Test that spectral PSNR is higher for less noisy images."""
    camera_image = normalise(camera()).astype(float)
    camera_image_with_noise_high = add_noise(camera_image)
    camera_image_with_noise_low = add_noise(
        camera_image, intensity=10, variance=0.1, sap=0.000001
    )

    ji_high = spectral_psnr(camera_image, camera_image_with_noise_high)
    ji_low = spectral_psnr(camera_image, camera_image_with_noise_low)

    assert ji_high > ji_low


def test_mutual_information():
    """Test that mutual information is higher for identical images than noisy pairs."""
    camera_image = camera()
    camera_image_with_noise = add_noise(camera())

    mi = mutual_information(camera_image, camera_image, normalised=False)
    mi_n = mutual_information(camera_image, camera_image_with_noise, normalised=False)

    assert mi > mi_n


def test_normalised_mutual_information():
    """Test that normalised mutual information equals 1 for identical images."""
    camera_image = camera()
    camera_image_with_noise = add_noise(camera())

    assert mutual_information(
        camera_image, camera_image, normalised=True
    ) == pytest.approx(1)
    assert mutual_information(
        camera_image_with_noise, camera_image_with_noise, normalised=True
    ) == pytest.approx(1)

    assert (
        mutual_information(camera_image, camera_image_with_noise, normalised=True) < 1
    )


def test_spectral_mutual_information():
    """Test that spectral mutual information decreases with added noise."""
    camera_image = camera()
    camera_image_with_noise = add_noise(camera())

    smi = spectral_mutual_information(camera_image, camera_image)
    smi_n = spectral_mutual_information(camera_image, camera_image_with_noise)

    assert smi_n < smi


def test_joint_information():
    """Test that joint information increases when noise is added."""
    camera_image = camera()
    camera_image_with_noise = add_noise(camera(), intensity=5, variance=3)

    ji = joint_information(camera_image, camera_image)
    ji_n = joint_information(camera_image, camera_image_with_noise)

    assert ji < ji_n


def test_spectral_psnr_identical_images():
    """Test spectral PSNR with identical images returns a high value."""
    camera_image = normalise(camera()).astype(float)
    spsnr = spectral_psnr(camera_image, camera_image)
    # PSNR for identical images should be very high (or inf)
    assert spsnr > 30 or spsnr == float('inf')


def test_mutual_information_symmetric():
    """Test that mutual information is symmetric."""
    camera_image = camera()
    camera_image_with_noise = add_noise(camera())

    mi_ab = mutual_information(camera_image, camera_image_with_noise, normalised=False)
    mi_ba = mutual_information(camera_image_with_noise, camera_image, normalised=False)
    # Mutual information should be approximately symmetric
    assert abs(mi_ab - mi_ba) < 0.1 * max(abs(mi_ab), abs(mi_ba), 1)
