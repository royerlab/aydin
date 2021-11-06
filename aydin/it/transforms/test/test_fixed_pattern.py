import numpy
from scipy.ndimage import gaussian_filter
from skimage.data import binary_blobs
from skimage.util import random_noise

from aydin.it.transforms.fixedpattern import FixedPatternTransform


def add_patterned_noise(image, n):
    image = image.copy()
    image *= 1 + 0.1 * (numpy.random.rand(n, n) - 0.5)
    image += 0.1 * numpy.random.rand(n, n)
    # image += 0.1*numpy.random.rand(n)[]
    image = random_noise(image, mode="gaussian", var=0.00001, seed=0)
    image = random_noise(image, mode="s&p", amount=0.000001, seed=0)
    return image


def test_fixed_pattern_real():
    n = 128
    image = binary_blobs(length=n, seed=1, n_dim=3, volume_fraction=0.01).astype(
        numpy.float32
    )
    image = gaussian_filter(image, sigma=4)
    noisy = add_patterned_noise(image, n).astype(numpy.float32)

    bs = FixedPatternTransform(sigma=0)

    preprocessed = bs.preprocess(noisy)
    postprocessed = bs.postprocess(preprocessed)

    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(image, name='image')
    #     viewer.add_image(noisy, name='noisy')
    #     viewer.add_image(preprocessed, name='preprocessed')
    #     viewer.add_image(postprocessed, name='postprocessed')

    assert image.shape == postprocessed.shape
    assert image.dtype == postprocessed.dtype
    assert numpy.abs(preprocessed - image).mean() < 0.007

    assert preprocessed.dtype == postprocessed.dtype
    assert numpy.abs(postprocessed - noisy).mean() < 1e-8

    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(image, name='image')
    #     viewer.add_image(noisy, name='noisy')
    #     viewer.add_image(corrected, name='corrected')
