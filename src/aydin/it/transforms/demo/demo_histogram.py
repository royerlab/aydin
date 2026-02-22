"""Demo of histogram equalisation transform.

Demonstrates the ``HistogramEqualisationTransform`` by applying it to
a camera image and verifying that postprocessing recovers the original
image.
"""

import numpy

from aydin.io.datasets import camera
from aydin.it.transforms.histogram import HistogramEqualisationTransform


def demo_histogram():
    """Apply histogram equalisation and verify roundtrip reconstruction."""
    image = camera()

    ht = HistogramEqualisationTransform()

    preprocessed = ht.preprocess(image)
    postprocessed = ht.postprocess(preprocessed)

    import napari

    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(preprocessed, name='preprocessed')
    viewer.add_image(postprocessed, name='postprocessed')
    napari.run()

    assert postprocessed.shape == image.shape
    assert numpy.abs(postprocessed - image).mean() < 1e-5


if __name__ == "__main__":
    demo_histogram()
