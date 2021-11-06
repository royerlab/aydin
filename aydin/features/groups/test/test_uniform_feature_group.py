import numpy
from skimage.exposure import rescale_intensity

from aydin.features.groups.uniform import UniformFeatures
from aydin.io.datasets import camera


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_uniform_feature_group():
    # get image:
    image = n(camera().astype(numpy.float32))

    # Instantiates median features:
    uniform = UniformFeatures(
        include_corner_features=True, include_scale_one=True, include_fine_features=True
    )

    for description in uniform._get_feature_descriptions_list(image.ndim):
        print(description)
    assert uniform.num_features(image.ndim) == 65

    # Check receptive field radius:
    assert uniform.receptive_field_radius == 3070

    # Set image:
    uniform.prepare(image)

    # compute features and check their valididty:
    feature = numpy.empty_like(image)
    for index in range(uniform.num_features(image.ndim)):
        uniform.compute_feature(index=index, feature=feature)
        assert (feature != image).any()
