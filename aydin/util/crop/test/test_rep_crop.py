# flake8: noqa

from aydin.io.datasets import newyork
from aydin.util.crop.demo.demo_rep_crop import demo_representative_crop
from aydin.util.crop.demo.demo_sf_rep_crop import demo_super_fast_representative_crop


def test_representative_crop():
    newyork_image = newyork()
    demo_representative_crop(newyork_image, display=False)


def test_super_fast_representative_crop():
    newyork_image = newyork()
    demo_super_fast_representative_crop(newyork_image, display=False)
