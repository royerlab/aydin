# flake8: noqa
from aydin.io.datasets import cropped_newyork
from aydin.util.j_invariance.demo.demo_j_invariant_classic import (
    demo_j_invariant_classic,
)


def test_j_invariant_classic():
    demo_j_invariant_classic(cropped_newyork(crop_amount=470), display=False)
