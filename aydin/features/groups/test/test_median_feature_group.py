import numpy
from scipy.ndimage import median_filter
from skimage.exposure import rescale_intensity

from aydin.features.groups.median import MedianFeatures
from aydin.io.datasets import camera


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_median_feature_group():
    # get image:
    image = n(camera().astype(numpy.float32))

    # Instantiates median features:
    radii = [1, 2, 4, 8]
    medians = MedianFeatures(radii=radii)
    assert medians.num_features(image.ndim) == 4

    # Check receptive field radius:
    assert medians.receptive_field_radius == 8

    # Set image:
    medians.prepare(image)

    # compute features and check their valididty:
    feature = numpy.empty_like(image)
    for index in range(medians.num_features(image.ndim)):
        medians.compute_feature(index=index, feature=feature)
        radius = radii[index]
        assert (feature == median_filter(image, size=2 * radius + 1)).all()
