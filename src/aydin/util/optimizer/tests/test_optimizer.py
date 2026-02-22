"""Tests for the Optimizer on standard mathematical test functions."""

# flake8: noqa
from aydin.util.optimizer.demo.demo_optimizer import (
    demo_optimizer_beale,
    demo_optimizer_booth,
    demo_optimizer_goldsteinprice,
    demo_optimizer_himmelblau,
    demo_optimizer_matyas,
    demo_optimizer_mccormik,
    demo_optimizer_rosenbrock,
    demo_optimizer_sphere,
)


def test_optimizer_beale():
    """Test optimizer convergence on the Beale function."""
    demo_optimizer_beale()


def test_optimizer_mccormik():
    """Test optimizer convergence on the McCormick function."""
    demo_optimizer_mccormik()


def test_optimizer_booth():
    """Test optimizer convergence on the Booth function."""
    demo_optimizer_booth()


def test_optimizer_sphere():
    """Test optimizer convergence on the Sphere function."""
    demo_optimizer_sphere()


def test_optimizer_rosenbrock():
    """Test optimizer convergence on the Rosenbrock function."""
    demo_optimizer_rosenbrock()


def test_optimizer_goldsteinprice():
    """Test optimizer convergence on the Goldstein-Price function."""
    demo_optimizer_goldsteinprice()


def test_optimizer_matyas():
    """Test optimizer convergence on the Matyas function."""
    demo_optimizer_matyas()


def test_optimizer_himmelblau():
    """Test optimizer convergence on the Himmelblau function."""
    demo_optimizer_himmelblau()
