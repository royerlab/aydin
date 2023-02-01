import numpy
from skimage.data import binary_blobs

from aydin.io.datasets import normalise
from aydin.it.transforms.padding import PaddingTransform


def test_padding():
    image = binary_blobs(length=128, seed=1, n_dim=3).astype(numpy.float32)
    image = normalise(image)

    pt = PaddingTransform(pad_width=17)

    preprocessed = pt.preprocess(image)
    postprocessed = pt.postprocess(preprocessed)

    assert postprocessed.dtype == image.dtype
    assert postprocessed.shape == image.shape
    assert numpy.abs(postprocessed - image).mean() < 1e-8
