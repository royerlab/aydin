"""Tests for the Noise2Self CNN denoising restoration API.

Tests the high-level :class:`Noise2SelfCNN` denoiser covering construction,
implementation discovery, configurable arguments, and description generation.
"""

from aydin.restoration.denoise.noise2selfcnn import Noise2SelfCNN


def test_n2s_cnn_implementations_discovery():
    """Test that implementations returns non-empty list with expected variants."""
    n2s = Noise2SelfCNN()
    impls = n2s.implementations
    assert isinstance(impls, list)
    assert len(impls) > 0
    for name in impls:
        assert name.startswith('Noise2SelfCNN-')
    # Expected models
    impl_names = [x.split('-', 1)[1] for x in impls]
    assert 'unet' in impl_names
    assert 'res_unet' in impl_names
    assert 'dncnn' in impl_names


def test_n2s_cnn_configurable_arguments():
    """Test that configurable_arguments returns valid structure for all models.

    This catches module-to-class name mismatches like res_unet -> ResidualUNetModel.
    """
    n2s = Noise2SelfCNN()
    args = n2s.configurable_arguments
    assert isinstance(args, dict)
    assert len(args) > 0
    for key, value in args.items():
        assert key.startswith('Noise2SelfCNN-')
        assert 'model' in value
        assert 'it' in value
        # Validate each sub-dict has expected keys
        for sub_key in ['model', 'it']:
            assert 'arguments' in value[sub_key]
            assert 'defaults' in value[sub_key]
            assert 'annotations' in value[sub_key]
            assert 'reference_class' in value[sub_key]


def test_n2s_cnn_implementations_description():
    """Test that implementations_description returns descriptions for all models.

    This catches class name lookup failures (e.g. res_unet -> ResidualUNetModel).
    """
    n2s = Noise2SelfCNN()
    descriptions = n2s.implementations_description
    impls = n2s.implementations
    assert len(descriptions) == len(impls)
    for desc in descriptions:
        assert isinstance(desc, str)
        assert len(desc) > 0


def test_n2s_cnn_get_translator_with_variant():
    """Test that each model variant produces a valid translator."""
    from aydin.it.cnn_torch import ImageTranslatorCNNTorch

    for variant in ['unet', 'res_unet', 'dncnn']:
        n2s = Noise2SelfCNN(variant=variant)
        translator = n2s.get_translator()
        assert isinstance(translator, ImageTranslatorCNNTorch)
