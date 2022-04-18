import numpy
from skimage.exposure import rescale_intensity

from aydin.features.groups.lowpass import LowPassFeatures
from aydin.io.datasets import camera


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_lowpass_feature_group():
    # get image:
    image = n(camera().astype(numpy.float32))

    # Instantiates low-pass features:
    lowpass = LowPassFeatures()
    assert lowpass.num_features(image.ndim) == 8

    # Check receptive field radius:
    assert lowpass.receptive_field_radius == 3

    # Set image:
    lowpass.prepare(image)

    # compute features and check their valididty:
    feature = numpy.empty_like(image)
    for index in range(lowpass.num_features(image.ndim)):
        lowpass.compute_feature(index=index, feature=feature)
        freq_cutoff = lowpass.freq_cutoffs[index]
        assert freq_cutoff > 0
