import numpy
from skimage.exposure import rescale_intensity

from aydin.features.groups.spatial import SpatialFeatures
from aydin.io.datasets import camera


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_spatial_feature_group():
    # get image:
    image = n(camera().astype(numpy.float32))

    # Instantiates spatial features:
    spatial = SpatialFeatures()
    assert spatial.num_features(image.ndim) == image.ndim

    # Check receptive field radius:
    assert spatial.receptive_field_radius == 0

    # Set image:
    spatial.prepare(image)

    # compute features and check their valididty:
    feature_y = numpy.empty_like(image)
    spatial.compute_feature(index=0, feature=feature_y)
    assert feature_y[0, 0] == feature_y[0, 511]
    assert feature_y[0, 0] < feature_y[511, 0]

    feature_x = numpy.empty_like(image)
    spatial.compute_feature(index=1, feature=feature_x)
    assert feature_x[0, 0] < feature_x[0, 511]
    assert feature_x[0, 0] == feature_x[511, 0]
