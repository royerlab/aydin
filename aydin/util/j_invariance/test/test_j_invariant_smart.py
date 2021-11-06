# flake8: noqa
from aydin.io.datasets import cropped_newyork
from aydin.util.j_invariance.demo.demo_j_invariant_smart import demo_j_invariant_smart


def test_j_invariant_smart():
    demo_j_invariant_smart(cropped_newyork(crop_amount=470), display=False)
