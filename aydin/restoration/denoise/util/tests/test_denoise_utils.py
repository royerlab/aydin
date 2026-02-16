"""Tests for denoiser discovery and instantiation utilities."""

from unittest.mock import MagicMock

import pytest

from aydin.restoration.denoise.util.denoise_utils import (
    get_denoiser_class_instance,
    get_list_of_denoiser_implementations,
    get_pretrained_denoiser_class_instance,
)


def test_get_list_of_denoiser_implementations():
    """Should return non-empty lists of implementations and descriptions."""
    implementations, descriptions = get_list_of_denoiser_implementations()
    assert len(implementations) > 0
    assert len(descriptions) > 0
    assert len(implementations) == len(descriptions)


def test_get_list_of_denoiser_implementations_has_all_families():
    """Should include Classic, CNN, and FGR denoiser families."""
    implementations, _ = get_list_of_denoiser_implementations()
    prefixes = {name.split('-')[0] for name in implementations}
    assert 'Classic' in prefixes
    assert 'Noise2SelfCNN' in prefixes
    assert 'Noise2SelfFGR' in prefixes


def test_get_denoiser_class_instance_classic():
    """Should create a Classic denoiser from variant string."""
    denoiser = get_denoiser_class_instance('Classic-butterworth')
    from aydin.restoration.denoise.classic import Classic

    assert isinstance(denoiser, Classic)


def test_get_denoiser_class_instance_fgr():
    """Should create a Noise2SelfFGR denoiser from variant string."""
    denoiser = get_denoiser_class_instance('Noise2SelfFGR-cb')
    from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR

    assert isinstance(denoiser, Noise2SelfFGR)


def test_get_denoiser_class_instance_cnn():
    """Should create a Noise2SelfCNN denoiser from variant string."""
    denoiser = get_denoiser_class_instance('Noise2SelfCNN-unet')
    from aydin.restoration.denoise.noise2selfcnn import Noise2SelfCNN

    assert isinstance(denoiser, Noise2SelfCNN)


def test_configurable_arguments_all_variants():
    """Every discovered variant should have valid configurable_arguments.

    This catches issues like:
    - Module-to-class name mismatches (e.g. res_unet vs ResidualUNetModel)
    - Missing type annotations causing KeyError in the GUI
    """
    implementations, _ = get_list_of_denoiser_implementations()
    for variant in implementations:
        instance = get_denoiser_class_instance(variant=variant)
        args = instance.configurable_arguments
        assert variant in args, f"{variant} not found in its own configurable_arguments"
        for component, sub_dict in args[variant].items():
            assert 'arguments' in sub_dict, f"{variant}/{component} missing 'arguments'"
            assert 'defaults' in sub_dict, f"{variant}/{component} missing 'defaults'"
            assert (
                'annotations' in sub_dict
            ), f"{variant}/{component} missing 'annotations'"
            assert (
                'reference_class' in sub_dict
            ), f"{variant}/{component} missing 'reference_class'"
            # annotations should be a dict (possibly empty for unannotated params)
            assert isinstance(
                sub_dict['annotations'], dict
            ), f"{variant}/{component} annotations is not a dict"
            # arguments and defaults must have same length
            assert len(sub_dict['arguments']) == len(
                sub_dict['defaults']
            ), f"{variant}/{component} arguments/defaults length mismatch"


def test_get_pretrained_denoiser_class_instance_fgr():
    """Should wrap an FGR translator in a Noise2SelfFGR denoiser."""
    mock_it = MagicMock()
    mock_it.__class__.__name__ = 'ImageTranslatorFGR'
    denoiser = get_pretrained_denoiser_class_instance(mock_it)
    from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR

    assert isinstance(denoiser, Noise2SelfFGR)
    assert denoiser.it is mock_it


def test_get_pretrained_denoiser_class_instance_unknown():
    """Should raise ValueError for unrecognized model type."""
    mock_it = MagicMock()
    mock_it.__class__.__name__ = 'UnknownModel'
    with pytest.raises(ValueError, match="not supported"):
        get_pretrained_denoiser_class_instance(mock_it)
