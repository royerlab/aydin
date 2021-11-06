import numpy

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import camera


def test_standard_features_type_conservation():

    _run_type_conservation_test(numpy.float16)
    _run_type_conservation_test(numpy.float32)
    _run_type_conservation_test(numpy.float64)
    _run_type_conservation_test(numpy.uint8)
    _run_type_conservation_test(numpy.uint16)
    _run_type_conservation_test(numpy.uint32)
    _run_type_conservation_test(numpy.uint64)


def _run_type_conservation_test(dtype, required_dtype=None):
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
