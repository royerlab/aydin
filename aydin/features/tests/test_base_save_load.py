"""Tests for save/load roundtrip on StandardFeatureGenerator."""

import numpy

from aydin.features.base import FeatureGeneratorBase
from aydin.features.standard_features import StandardFeatureGenerator


def test_save_load_roundtrip(tmp_path):
    """Test that a StandardFeatureGenerator survives a save/load roundtrip."""
    # Create a feature generator with non-default settings:
    original = StandardFeatureGenerator(
        include_spatial_features=True,
        include_median_features=True,
        include_lowpass_features=True,
        num_lowpass_features=4,
        include_dct_features=False,
        include_random_conv_features=False,
        include_scale_one=True,
        include_fine_features=True,
        include_corner_features=False,
        max_level=10,
    )

    # Save to temporary directory:
    save_dir = str(tmp_path / "feature_gen")
    original.save(save_dir)

    # Load back:
    loaded = FeatureGeneratorBase.load(save_dir)

    # Verify the loaded object is the right type:
    assert isinstance(loaded, StandardFeatureGenerator)

    # Verify feature counts match for a 2D image:
    ndim = 2
    assert loaded.get_num_features(ndim) == original.get_num_features(ndim)

    # Verify receptive field radius matches:
    assert loaded.get_receptive_field_radius() == original.get_receptive_field_radius()

    # Verify that both generators produce the same features on an image:
    image = numpy.random.RandomState(42).rand(1, 1, 64, 64).astype(numpy.float32)
    features_original = original.compute(image, exclude_center_feature=True)
    features_loaded = loaded.compute(image, exclude_center_feature=True)

    assert features_original.shape == features_loaded.shape
    assert numpy.allclose(features_original, features_loaded)
