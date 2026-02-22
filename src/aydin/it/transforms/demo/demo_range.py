"""Demo of the range normalisation transform.

Demonstrates the ``RangeTransform`` in different modes (minmax and
percentile), verifying that postprocessing recovers the original image.
"""

import numpy

from aydin.io.datasets import newyork
from aydin.it.transforms.range import RangeTransform


def demo_range(mode):
    """Apply range normalisation and verify roundtrip reconstruction.

    Parameters
    ----------
    mode : str
        Normalisation mode, e.g. ``'minmax'`` or ``'percentile'``.
    """
    image = newyork()

    rt = RangeTransform(mode=mode)

    preprocessed = rt.preprocess(image)
    postprocessed = rt.postprocess(preprocessed)

    import napari

    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(preprocessed, name='preprocessed')
    viewer.add_image(postprocessed, name='postprocessed')
    napari.run()
    assert postprocessed.shape == image.shape
    assert numpy.abs(postprocessed - image).mean() < 1e-8


if __name__ == "__main__":
    demo_range("minmax")
    demo_range("percentile")
