"""Tests for signal-to-noise ratio estimation."""

from aydin.analysis.demo.demo_snr_estimate import demo_snr_estimate


def test_snr_estimate():
    """Test SNR estimation via the demo function."""
    demo_snr_estimate(display=False, run_as_demo=False)
