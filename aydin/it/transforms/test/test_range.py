import numpy
from skimage.data import binary_blobs

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

    assert postprocessed.dtype == image.dtype
    assert postprocessed.shape == image.shape
    assert numpy.abs(postprocessed - image).mean() < 1e-8


def test_range_3d():
    """Test range transform on 3D volume."""
    image = binary_blobs(length=32, n_dim=3, rng=1).astype(numpy.float32)

    for mode in ["minmax", "percentile"]:
        rt = RangeTransform(mode=mode)

        preprocessed = rt.preprocess(image)
        postprocessed = rt.postprocess(preprocessed)

        assert preprocessed.shape == image.shape
        assert postprocessed.dtype == image.dtype
        assert postprocessed.shape == image.shape
        assert numpy.abs(postprocessed - image).mean() < 1e-8
