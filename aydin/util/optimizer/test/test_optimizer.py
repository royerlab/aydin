# flake8: noqa
from aydin.util.optimizer.demo.demo_optimizer import (
    demo_optimizer_beale,
    demo_optimizer_mccormik,
    demo_optimizer_booth,
    demo_optimizer_sphere,
    demo_optimizer_rosenbrock,
    demo_optimizer_goldsteinprice,
    demo_optimizer_matyas,
    demo_optimizer_himmelblau,
)


def test_optimizer_beale():
    demo_optimizer_beale()


def test_optimizer_mccormik():
    demo_optimizer_mccormik()


def test_optimizer_booth():
    demo_optimizer_booth()


def test_optimizer_sphere():
    demo_optimizer_sphere()


def test_optimizer_rosenbrock():
    demo_optimizer_rosenbrock()


def test_optimizer_goldsteinprice():
    demo_optimizer_goldsteinprice()


def test_optimizer_matyas():
    demo_optimizer_matyas()


def test_optimizer_himmelblau():
    demo_optimizer_himmelblau()
