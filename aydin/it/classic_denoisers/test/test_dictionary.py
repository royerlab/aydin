import pytest

from aydin.io.datasets import cropped_newyork
from aydin.it.classic_denoisers.demo.demo_2D_dictionary_fixed import (
    demo_dictionary_fixed,
)
from aydin.it.classic_denoisers.demo.demo_2D_dictionary_learned import (
    demo_dictionary_learned,
)
from aydin.it.classic_denoisers.dictionary_fixed import denoise_dictionary_fixed

from aydin.it.classic_denoisers.test.util_test_nd import check_nd


def test_dictionary_learned():
    assert demo_dictionary_learned(cropped_newyork(), display=False) >= 0.636 - 0.01


@pytest.mark.heavy
def test_dictionary_fixed():
    assert demo_dictionary_fixed(cropped_newyork(), display=False) >= 0.636 - 0.01


def test_dictionary_nd():
    check_nd(denoise_dictionary_fixed)
