import numpy
from skimage.util import random_noise

from aydin.io.datasets import normalise, camera
from aydin.it.transforms.highpass import HighpassTransform


def test_high_pass():
    image = normalise(camera())
    image = random_noise(image, mode="s&p", amount=0.03, seed=0, clip=False)

    ac = HighpassTransform()

    preprocessed = ac.preprocess(image)
    postprocessed = ac.postprocess(preprocessed)

    assert image.shape == postprocessed.shape
    assert image.dtype == postprocessed.dtype

    assert postprocessed.dtype == image.dtype
    assert numpy.abs(postprocessed - image).mean() < 1e-8
