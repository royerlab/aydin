# flake8: noqa
from aydin.io.datasets import cropped_newyork
from aydin.util.j_invariance.demo.demo_j_invariant_only_cat import (
    demo_j_invariant_only_cat,
)


def test_j_invariant_only_cat():
    demo_j_invariant_only_cat(cropped_newyork(crop_amount=470), display=False)
