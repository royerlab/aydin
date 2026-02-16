"""Tests for camera noise simulation."""

import numpy as np
from numpy.random import RandomState

from aydin.analysis.camera_simulation import simulate_camera_image


def test_output_shape():
    """Output should have the same shape as input."""
    photons = np.full((64, 64), 100.0, dtype=np.float32)
    result = simulate_camera_image(photons)
    assert result.shape == (64, 64)


def test_output_dtype():
    """Output should have the requested integer dtype."""
    photons = np.full((32, 32), 100.0, dtype=np.float32)
    result = simulate_camera_image(photons, dtype=np.int32)
    assert result.dtype == np.int32

    result16 = simulate_camera_image(photons, dtype=np.int16)
    assert result16.dtype == np.int16


def test_reproducibility():
    """Same random seeds should produce identical results."""
    photons = np.full((32, 32), 500.0, dtype=np.float32)
    r1 = simulate_camera_image(
        photons, shot_rnd=RandomState(0), camera_rnd=RandomState(42)
    )
    r2 = simulate_camera_image(
        photons, shot_rnd=RandomState(0), camera_rnd=RandomState(42)
    )
    np.testing.assert_array_equal(r1, r2)


def test_value_range():
    """All values should be non-negative and at most 2^bitdepth - 1."""
    photons = np.full((32, 32), 500.0, dtype=np.float32)
    bitdepth = 12
    result = simulate_camera_image(photons, bitdepth=bitdepth, shot_rnd=RandomState(0))
    assert result.min() >= 0
    assert result.max() <= 2**bitdepth - 1


def test_saturation():
    """Very high photon counts should saturate at 2^bitdepth - 1."""
    photons = np.full((32, 32), 1e8, dtype=np.float32)
    bitdepth = 12
    max_adu = 2**bitdepth - 1
    result = simulate_camera_image(photons, bitdepth=bitdepth, shot_rnd=RandomState(0))
    assert result.max() == max_adu


def test_baseline_offset():
    """Zero-photon image should have values around the baseline."""
    photons = np.zeros((32, 32), dtype=np.float32)
    baseline = 100
    result = simulate_camera_image(
        photons, baseline=baseline, shot_rnd=RandomState(0), camera_rnd=RandomState(42)
    )
    # Most values should be near baseline (offset + dark current is small)
    assert np.median(result) >= baseline * 0.5
