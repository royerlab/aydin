# flake8: noqa
from aydin.io.datasets import cropped_newyork
from aydin.util.j_invariance.demo.demo_j_invariant import demo_j_invariant


def test_j_invariant():
    demo_j_invariant(
        cropped_newyork(crop_amount=470), optimiser_mode='fast', display=False
    )
    demo_j_invariant(
        cropped_newyork(crop_amount=470), optimiser_mode='smart', display=False
    )
