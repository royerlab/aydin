"""Tests for image resolution estimation."""

from aydin.analysis.demo.demo_resolution_estimate import demo_resolution_estimate


def test_resolution_estimate():
    """Test resolution estimation via the demo function."""
    demo_resolution_estimate(display=False)
