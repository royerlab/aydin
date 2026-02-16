"""Tests for the LowPassFeatures feature group."""

import numpy

from aydin.features.groups.lowpass import LowPassFeatures


def test_lowpass_feature_group(normalized_camera_image):
    """Test low-pass feature count, receptive field, and frequency cutoffs."""
    # get image:
    image = normalized_camera_image

    # settings:
    num_features = 9
    max_size = 11

    # Instantiates low-pass features:
    lowpass = LowPassFeatures(num_features=num_features, max_size=max_size)
    assert lowpass.num_features(image.ndim) == num_features

    # Check receptive field radius:
    assert lowpass.receptive_field_radius == max_size // 2

    # Set image:
    lowpass.prepare(image)

    # compute features and check their validity:
    feature = numpy.empty_like(image)
    for index in range(lowpass.num_features(image.ndim)):
        lowpass.compute_feature(index=index, feature=feature)
        freq_cutoff = lowpass.freq_cutoffs[index]
        assert freq_cutoff > 0

    lowpass.finish()
