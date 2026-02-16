"""Tests for the fast edge filter."""

from aydin.util.edge_filter.demo.demo import demo_fast_edge


def test_fast_edge_filter():
    """Test fast edge filter demo runs without errors."""
    demo_fast_edge(display=False)
