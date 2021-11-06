import numpy

from aydin.io.datasets import newyork
from aydin.it.transforms.range import RangeTransform


def demo_range(mode):
    image = newyork()

    rt = RangeTransform(mode=mode)

    preprocessed = rt.preprocess(image)
    postprocessed = rt.postprocess(preprocessed)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(preprocessed, name='preprocessed')
        viewer.add_image(postprocessed, name='postprocessed')

    assert postprocessed.shape == image.shape
    assert numpy.abs(postprocessed - image).mean() < 1e-8


demo_range("minmax")
demo_range("percentile")
