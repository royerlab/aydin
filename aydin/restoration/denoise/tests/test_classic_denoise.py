"""Tests for the Classic denoising restoration API.

Tests the high-level :class:`Classic` denoiser and the :func:`classic_denoise`
convenience function, covering construction, implementation discovery,
configurable arguments, train/denoise cycle, and custom transforms.
"""

import numpy
import pytest
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr

from aydin.io.datasets import add_noise, normalise
from aydin.it.transforms.range import RangeTransform
from aydin.restoration.denoise.classic import Classic, classic_denoise


def _make_noisy_image():
    """Return a normalised float32 camera image with added noise."""
    image = normalise(camera().astype(numpy.float32))
    noisy = add_noise(image)
    return image, noisy


def test_classic_implementations_discovery():
    """Test that implementations returns a non-empty list of Classic- prefixed names."""
    classic = Classic()
    impls = classic.implementations
    assert isinstance(impls, list)
    assert len(impls) > 0
    for name in impls:
        assert name.startswith('Classic-')
    # Common variants should be present
    impl_names = [x.split('-', 1)[1] for x in impls]
    assert 'butterworth' in impl_names
    assert 'gaussian' in impl_names


def test_classic_configurable_arguments():
    """Test that configurable_arguments returns valid structure."""
    classic = Classic()
    args = classic.configurable_arguments
    assert isinstance(args, dict)
    assert len(args) > 0
    for key, value in args.items():
        assert key.startswith('Classic-')
        assert 'calibration' in value
        assert 'it' in value
        assert 'arguments' in value['calibration']
        assert 'defaults' in value['calibration']
        assert 'arguments' in value['it']
        assert 'defaults' in value['it']


def test_classic_implementations_description():
    """Test that implementations_description returns descriptions for each variant."""
    classic = Classic()
    descriptions = classic.implementations_description
    impls = classic.implementations
    assert len(descriptions) == len(impls)
    for desc in descriptions:
        assert isinstance(desc, str)
        assert len(desc) > 0


def test_classic_train_denoise_gaussian():
    """Test full train+denoise cycle with gaussian variant."""
    image, noisy = _make_noisy_image()
    transforms = [{"class": RangeTransform, "kwargs": {}}]

    classic = Classic(variant='gaussian', it_transforms=transforms)
    classic.train(noisy)
    denoised = classic.denoise(noisy)

    denoised = denoised.clip(0, 1)
    assert denoised.shape == image.shape
    assert denoised.dtype == noisy.dtype
    # Denoised should be better than noisy
    assert psnr(image, denoised) > psnr(image, noisy.clip(0, 1))


def test_classic_train_denoise_butterworth():
    """Test full train+denoise cycle with butterworth variant."""
    image, noisy = _make_noisy_image()
    transforms = [{"class": RangeTransform, "kwargs": {}}]

    classic = Classic(variant='butterworth', it_transforms=transforms)
    classic.train(noisy)
    denoised = classic.denoise(noisy)

    denoised = denoised.clip(0, 1)
    assert denoised.shape == image.shape
    assert psnr(image, denoised) > psnr(image, noisy.clip(0, 1))


def test_classic_with_custom_transforms():
    """Test Classic denoiser with custom transforms list."""
    _, noisy = _make_noisy_image()
    custom_transforms = [{"class": RangeTransform, "kwargs": {"mode": "percentile"}}]

    classic = Classic(variant='gaussian', it_transforms=custom_transforms)
    classic.train(noisy)
    denoised = classic.denoise(noisy)

    assert denoised.shape == noisy.shape


def test_classic_with_empty_transforms():
    """Test Classic denoiser with no transforms."""
    _, noisy = _make_noisy_image()

    classic = Classic(variant='gaussian', it_transforms=[])
    classic.train(noisy)
    denoised = classic.denoise(noisy)

    assert denoised.shape == noisy.shape


def test_classic_repr():
    """Test string representation of Classic denoiser."""
    classic = Classic(variant='gaussian')
    repr_str = repr(classic)
    assert 'Classic' in repr_str


def test_classic_get_translator_with_variant():
    """Test that get_translator returns an ImageDenoiserClassic with the set variant."""
    classic = Classic(variant='butterworth')
    translator = classic.get_translator()
    assert translator is not None


def test_classic_default_transforms():
    """Test that default transforms include Range, Padding, and VST."""
    classic = Classic()
    assert len(classic.it_transforms) == 3


def test_classic_custom_transforms_override():
    """Test that providing it_transforms replaces the defaults."""
    transforms = [{"class": RangeTransform, "kwargs": {}}]
    classic = Classic(it_transforms=transforms)
    assert len(classic.it_transforms) == 1
    assert classic.it_transforms[0]["class"] is RangeTransform


def test_classic_empty_transforms():
    """Test that empty transforms list results in no transforms."""
    classic = Classic(it_transforms=[])
    assert len(classic.it_transforms) == 0


def test_classic_constructor_with_variant():
    """Test constructor stores variant correctly."""
    classic = Classic(variant='butterworth')
    assert classic.variant == 'butterworth'


def test_classic_constructor_with_lower_level_args():
    """Test constructor stores lower_level_args."""
    args = {"variant": "Classic-gaussian", "calibration": {}, "processing": None}
    classic = Classic(lower_level_args=args)
    assert classic.lower_level_args is args


def test_classic_constructor_with_model_path():
    """Test constructor stores model path and flag."""
    classic = Classic(use_model=True, input_model_path="/tmp/model.zip")
    assert classic.use_model_flag is True
    assert classic.input_model_path == "/tmp/model.zip"


def test_classic_disabled_modules():
    """Test that disabled_modules contains expected entries."""
    assert "bilateral" in Classic.disabled_modules
    assert "bmnd" in Classic.disabled_modules
    assert "_defaults" in Classic.disabled_modules


def test_classic_implementations_do_not_include_disabled():
    """Disabled modules should not appear in implementations list."""
    classic = Classic()
    impls = classic.implementations
    for disabled in Classic.disabled_modules:
        assert f'Classic-{disabled}' not in impls


def test_classic_get_translator_no_variant():
    """get_translator() without variant returns default ImageDenoiserClassic."""
    from aydin.it.classic import ImageDenoiserClassic

    classic = Classic()
    translator = classic.get_translator()
    assert isinstance(translator, ImageDenoiserClassic)


def test_classic_add_transforms_to_translator():
    """add_transforms() should add transforms to the translator."""
    transforms = [{"class": RangeTransform, "kwargs": {}}]
    classic = Classic(variant='gaussian', it_transforms=transforms)
    classic.it = classic.get_translator()
    classic.add_transforms()
    assert len(classic.it.transforms_list) == 1


@pytest.mark.heavy
def test_classic_denoise_convenience_function():
    """Test the module-level classic_denoise() convenience function."""
    image, noisy = _make_noisy_image()
    denoised = classic_denoise(noisy, variant='gaussian')

    denoised = denoised.clip(0, 1)
    assert denoised.shape == image.shape
    assert psnr(image, denoised) > psnr(image, noisy.clip(0, 1))
