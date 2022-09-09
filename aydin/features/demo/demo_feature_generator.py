# flake8: noqa
import numpy

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import pollen


def demo_collect_feature_2d():
    generator = StandardFeatureGenerator(
        include_corner_features=True,
        include_scale_one=False,
        include_fine_features=True,
        include_spatial_features=True,
        # include_median_features=True,
        # include_dct_features=True,
        # num_sinusoidal_features=4,
        # include_random_conv_features=True,
    )

    image = numpy.zeros(shape=(17, 17), dtype=numpy.float32)
    image[8, 8] = 1
    image[::2, ::2] += 0.05

    # feature generator requires images in 'standard' form: BCTZYX
    norm_image = image[numpy.newaxis, numpy.newaxis, ...]

    features = generator.compute(
        norm_image,
        exclude_center_value=True,
        spatial_feature_offset=(0, 0),
        spatial_feature_scale=(1 / 17, 1 / 17),
    )

    assert features is not None

    features = features.squeeze()

    features = numpy.transpose(features, axes=(2, 0, 1))

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image', colormap='plasma')
        viewer.add_image(features, name='features', colormap='plasma')


def demo_collect_feature_3d():
    generator = StandardFeatureGenerator(
        include_corner_features=True,
        include_scale_one=False,
        include_fine_features=True,
        include_spatial_features=True,
        # include_median_features=True,
        # include_dct_features=True,
        # num_sinusoidal_features=4,
        # include_random_conv_features=True,
    )

    image = numpy.zeros(shape=(17, 17, 17), dtype=numpy.float32)
    image[8, 8, 8] = 1
    image[::2, ::2, ::2] += 0.05

    # feature generator requires images in 'standard' form: BCTZYX
    norm_image = image[numpy.newaxis, numpy.newaxis, ...]

    features = generator.compute(
        norm_image,
        exclude_center_value=True,
        spatial_feature_offset=(0, 0, 0),
        spatial_feature_scale=(1 / 17, 1 / 17, 1 / 17),
    )

    assert features is not None

    features = features.squeeze()

    features = numpy.transpose(features, axes=(3, 0, 1, 2))

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image', colormap='plasma')
        viewer.add_image(features, name='features', colormap='plasma')


def demo_collect_feature_2d_image():
    generator = StandardFeatureGenerator(
        include_corner_features=True,
        include_scale_one=False,
        include_fine_features=True,
        include_spatial_features=True,
        # include_median_features=True,
        # include_dct_features=True,
        # num_sinusoidal_features=4,
        # include_random_conv_features=True,
    )

    image = pollen()

    # feature generator requires images in 'standard' form: BCTZYX
    norm_image = image[numpy.newaxis, numpy.newaxis, ...]

    features = generator.compute(
        norm_image,
        exclude_center_value=True,
        spatial_feature_offset=(0, 0),
        spatial_feature_scale=(1 / 17, 1 / 17),
    )

    assert features is not None

    features = features.squeeze()

    features = numpy.transpose(features, axes=(2, 0, 1))

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image', colormap='plasma')
        viewer.add_image(features, name='features', colormap='plasma')


if __name__ == "__main__":
    demo_collect_feature_3d()
    demo_collect_feature_2d()

    demo_collect_feature_2d_image()
