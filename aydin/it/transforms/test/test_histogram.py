import numpy

from aydin.io.datasets import camera
from aydin.it.transforms.histogram import HistogramEqualisationTransform


def test_histogram():
    image = camera()

    ht = HistogramEqualisationTransform()

    preprocessed = ht.preprocess(image)
    postprocessed = ht.postprocess(preprocessed)

    # import napari
    #
    # viewer = napari.Viewer()
    # viewer.add_image(image, name='image')
    # viewer.add_image(preprocessed, name='preprocessed')
    # viewer.add_image(postprocessed, name='postprocessed')
    # napari.run()

    assert postprocessed.dtype == image.dtype
    assert postprocessed.shape == image.shape
    assert (
        numpy.abs(
            postprocessed.astype(numpy.float32) - image.astype(numpy.float32)
        ).mean()
        < 2
    )
