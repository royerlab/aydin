import numpy

from aydin.io.datasets import newyork
from aydin.it.transforms.range import RangeTransform


def test_range_minmax():
    do_test_range("minmax")


def test_range_percentile():
    do_test_range("percentile")


def do_test_range(mode):
    image = newyork()

    rt = RangeTransform(mode=mode)

    preprocessed = rt.preprocess(image)
    postprocessed = rt.postprocess(preprocessed)

    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(image, name='image')
    #     viewer.add_image(preprocessed, name='preprocessed')
    #     viewer.add_image(postprocessed, name='postprocessed')

    assert postprocessed.dtype == image.dtype
    assert postprocessed.shape == image.shape
    assert numpy.abs(postprocessed - image).mean() < 1e-8
