"""Tests for transform save/load round-trips via jsonpickle.

Each test verifies that a transform can be serialized (jsonpickle.encode),
deserialized (jsonpickle.decode), and still produce identical preprocess
and postprocess results. This catches silent state loss from __getstate__
excluding essential fields.
"""

import jsonpickle
import numpy
import numpy.testing
import pytest

from aydin.it.transforms.fixedpattern import FixedPatternTransform
from aydin.it.transforms.highpass import HighpassTransform
from aydin.it.transforms.histogram import HistogramEqualisationTransform
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform


@pytest.fixture
def test_image_2d():
    """A reproducible 2D float32 test image with realistic value range."""
    rng = numpy.random.RandomState(42)
    return rng.uniform(100, 1000, (64, 64)).astype(numpy.float32)


@pytest.fixture
def test_image_3d():
    """A reproducible 3D float32 test image for transforms needing >2 dims."""
    rng = numpy.random.RandomState(42)
    return rng.uniform(100, 1000, (8, 64, 64)).astype(numpy.float32)


def _roundtrip(transform, image):
    """Helper: preprocess, serialize, deserialize, compare pre+postprocess.

    Returns the preprocessed arrays from original and loaded transforms.
    """
    preprocessed = transform.preprocess(image)

    encoded = jsonpickle.encode(transform)
    loaded = jsonpickle.decode(encoded)

    preprocessed_loaded = loaded.preprocess(image)
    numpy.testing.assert_array_almost_equal(
        preprocessed, preprocessed_loaded, decimal=5
    )

    postprocessed = transform.postprocess(preprocessed.copy())
    postprocessed_loaded = loaded.postprocess(preprocessed_loaded.copy())
    numpy.testing.assert_array_almost_equal(
        postprocessed, postprocessed_loaded, decimal=5
    )

    return preprocessed, preprocessed_loaded


def test_range_saveload(test_image_2d):
    """RangeTransform: normaliser, min/max excluded by __getstate__."""
    t = RangeTransform(mode='percentile', percentile=0.99)
    _roundtrip(t, test_image_2d)


def test_range_saveload_minmax(test_image_2d):
    """RangeTransform with default minmax mode."""
    t = RangeTransform(mode='minmax')
    _roundtrip(t, test_image_2d)


def test_padding_saveload(test_image_2d):
    """PaddingTransform: pad_width excluded by __getstate__."""
    t = PaddingTransform(pad_width=5, mode='reflect')
    preprocessed, preprocessed_loaded = _roundtrip(t, test_image_2d)

    # Verify padding actually changed the shape
    assert preprocessed.shape[0] > test_image_2d.shape[0]
    assert preprocessed.shape[1] > test_image_2d.shape[1]


def test_padding_saveload_constant(test_image_2d):
    """PaddingTransform with constant padding mode."""
    t = PaddingTransform(pad_width=3, mode='constant')
    _roundtrip(t, test_image_2d)


def test_highpass_saveload(test_image_2d):
    """HighpassTransform: low_pass_image, dtype, min/max excluded."""
    t = HighpassTransform(sigma=2.0, median_filtering=False)
    _roundtrip(t, test_image_2d)


def test_highpass_saveload_median(test_image_2d):
    """HighpassTransform with median filtering enabled."""
    t = HighpassTransform(sigma=1.0, median_filtering=True)
    _roundtrip(t, test_image_2d)


def test_fixedpattern_saveload(test_image_3d):
    """FixedPatternTransform: corrections dict excluded by __getstate__."""
    t = FixedPatternTransform(percentile=5.0, sigma=1.0)
    _roundtrip(t, test_image_3d)


def test_histogram_equalize_saveload(test_image_2d):
    """HistogramEqualisationTransform (equalize): cdf, bin_centers excluded."""
    t = HistogramEqualisationTransform(mode='equalize')
    _roundtrip(t, test_image_2d)


def test_vst_anscomb_saveload(test_image_2d):
    """VarianceStabilisationTransform (anscomb): dtype, min/max, transform excluded."""
    t = VarianceStabilisationTransform(mode='anscomb')
    _roundtrip(t, test_image_2d)


def test_vst_sqrt_saveload(test_image_2d):
    """VarianceStabilisationTransform (sqrt mode)."""
    t = VarianceStabilisationTransform(mode='sqrt')
    _roundtrip(t, test_image_2d)
