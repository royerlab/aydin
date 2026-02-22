"""Demo of the padding transform for image boundary handling.

Demonstrates the ``PaddingTransform`` by applying padding to a 3D
binary blobs image and verifying that postprocessing recovers the
original shape and values.
"""

import numpy
from skimage.data import binary_blobs

from aydin.io.datasets import normalise
from aydin.it.transforms.padding import PaddingTransform


def demo_padding():
    """Apply padding to a 3D image and verify roundtrip reconstruction."""
    image = binary_blobs(length=128, rng=1, n_dim=3).astype(numpy.float32)
    image = normalise(image)

    pt = PaddingTransform(pad_width=17)

    preprocessed = pt.preprocess(image)
    postprocessed = pt.postprocess(preprocessed)

    import napari

    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(preprocessed, name='preprocessed')
    viewer.add_image(postprocessed, name='postprocessed')
    napari.run()
    assert postprocessed.shape == image.shape
    assert numpy.abs(postprocessed - image).mean() < 1e-8


if __name__ == "__main__":
    demo_padding()
