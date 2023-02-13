import numpy
import pytest

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import camera


@pytest.mark.parametrize(
    "dtype",
    [
        numpy.float16,
        numpy.float32,
        numpy.float64,
        numpy.uint8,
        numpy.uint16,
        numpy.uint32,
        numpy.uint64,
    ],
)
def test_standard_features_type_conservation_test(dtype, required_dtype=None):
    image = camera().astype(dtype)
    # feature generator requires images in 'standard' form: BCTZYX
    image = image[numpy.newaxis, numpy.newaxis, ...]
    feature_gen = StandardFeatureGenerator(dtype=required_dtype)
    features = feature_gen.compute(image, exclude_center_feature=True)

    # float16 is a nasty case:
    if image.dtype == numpy.float16:
        assert features.dtype == numpy.float32
    else:
        assert features.dtype == image.dtype
