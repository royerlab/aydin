import numpy

from aydin.io.datasets import camera
from aydin.it.transforms.histogram import HistogramEqualisationTransform


def test_histogram():
    image = camera()

    ht = HistogramEqualisationTransform()

    preprocessed = ht.preprocess(image)
    postprocessed = ht.postprocess(preprocessed)

    assert postprocessed.dtype == image.dtype
    assert postprocessed.shape == image.shape
    assert (
        numpy.abs(
            postprocessed.astype(numpy.float32) - image.astype(numpy.float32)
        ).mean()
        < 2
    )
