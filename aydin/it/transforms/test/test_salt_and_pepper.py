# flake8: noqa
import numpy
from numpy.linalg import norm
from scipy.ndimage import median_filter
from skimage.util import random_noise

from aydin.io.datasets import camera, normalise, newyork
from aydin.it.transforms.salt_pepper import SaltPepperTransform


def test_suppress_fixed_background_real():
    image = normalise(newyork())
    noisy = random_noise(image, mode="s&p", amount=0.03, seed=0, clip=False)

    bpc = SaltPepperTransform(threshold=0.15)

    corrected = bpc.preprocess(noisy)

    median = median_filter(image, size=3)

    # import napari

    # with napari.gui_qt():
    # viewer = napari.Viewer()
    # viewer.add_image(image, name='image')
    # viewer.add_image(median, name='median')
    # viewer.add_image(corrected, name='corrected')

    error0 = numpy.abs(median - image).mean()
    error = numpy.abs(corrected - image).mean()

    print(f"Error noisy = {error0}")
    print(f"Error = {error}")
    assert error < 0.03
    assert error0 > error
