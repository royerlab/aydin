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

    # compute features and check their valididty:
    feature = numpy.empty_like(image)
    for index in range(lowpass.num_features(image.ndim)):
        lowpass.compute_feature(index=index, feature=feature)
        freq_cutoff = lowpass.freq_cutoffs[index]
        assert freq_cutoff > 0
