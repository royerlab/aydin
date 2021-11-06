# flake8: noqa
import numpy
import numpy as np
from skimage.data import camera

from aydin.io.datasets import normalise, add_noise, dots, lizard, pollen, newyork
from aydin.util.crop.demo.demo_rep_crop import demo_representative_crop
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.log.log import Log


def test_representative_crop():
    newyork_image = newyork()
    demo_representative_crop(newyork_image, display=False)
