"""Tests for correlation analysis functions."""

import numpy as np

from aydin.analysis.correlation import correlation, correlation_distance


def test_correlation_returns_tuple():
    """correlation() should return a tuple with one entry per dimension."""
    image = np.random.RandomState(0).randn(64, 64).astype(np.float32)
    result = correlation(image, nb_samples=256)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_autocorrelation_starts_near_one():
    """Autocorrelation at lag 0 should be normalized to 1.0."""
    image = np.random.RandomState(0).randn(128).astype(np.float32)
    result = correlation(image, nb_samples=512)
    assert result[0] is not None
    np.testing.assert_almost_equal(result[0][0], 1.0, decimal=5)


def test_short_dimension_returns_none():
    """Dimensions shorter than 3 should return None."""
    image = np.random.RandomState(0).randn(2, 64).astype(np.float32)
    result = correlation(image, nb_samples=256)
    assert result[0] is None  # dim 0 has length 2
    assert result[1] is not None  # dim 1 has length 64


def test_correlation_distance_zerocross():
    """Test zero-crossing method on correlated signal."""
    # Create a signal with known structure
    rng = np.random.RandomState(42)
    x = np.linspace(0, 8 * np.pi, 256)
    signal = np.sin(x) + 0.1 * rng.randn(256)
    signal = signal.astype(np.float32)
    dists = correlation_distance(signal, method='zerocross')
    assert isinstance(dists, tuple)
    assert len(dists) == 1
    assert dists[0] >= 0


def test_correlation_distance_firstmin():
    """Test first-minimum method returns non-negative distance."""
    rng = np.random.RandomState(42)
    image = rng.randn(64, 64).astype(np.float32)
    dists = correlation_distance(image, method='firstmin')
    assert isinstance(dists, tuple)
    assert len(dists) == 2
    assert all(d >= 0 for d in dists)


def test_cross_correlation():
    """Cross-correlation with a shifted copy should still work."""
    rng = np.random.RandomState(0)
    image = rng.randn(64, 64).astype(np.float32)
    shifted = np.roll(image, 2, axis=1)
    result = correlation(image, shifted, nb_samples=256)
    assert isinstance(result, tuple)
    assert len(result) == 2
