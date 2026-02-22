"""Tests for the Noise2Self FGR denoising restoration API.

Tests the high-level :class:`Noise2SelfFGR` denoiser and the
:func:`noise2self_fgr` convenience function, covering construction,
implementation discovery, configurable arguments, and train/denoise cycles.
"""

import numpy
import pytest
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr

from aydin.io.datasets import add_noise, normalise
from aydin.it.transforms.range import RangeTransform
from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR, noise2self_fgr


def _make_noisy_image():
    """Return a normalised float32 camera image with added noise."""
    image = normalise(camera().astype(numpy.float32))
    noisy = add_noise(image)
    return image, noisy


def test_n2s_fgr_implementations_discovery():
    """Test that implementations returns non-empty list with expected variants."""
    n2s = Noise2SelfFGR()
    impls = n2s.implementations
    assert isinstance(impls, list)
    assert len(impls) > 0
    for name in impls:
        assert name.startswith('Noise2SelfFGR-')
    # Expected regressors (lgbm may be absent if libomp is missing)
    impl_names = [x.split('-', 1)[1] for x in impls]
    assert 'cb' in impl_names
    assert 'linear' in impl_names


def test_n2s_fgr_configurable_arguments():
    """Test that configurable_arguments returns valid structure."""
    n2s = Noise2SelfFGR()
    args = n2s.configurable_arguments
    assert isinstance(args, dict)
    assert len(args) > 0
    for key, value in args.items():
        assert key.startswith('Noise2SelfFGR-')
        assert 'feature_generator' in value
        assert 'regressor' in value
        assert 'it' in value
        # Validate each sub-dict has expected keys
        for sub_key in ['feature_generator', 'regressor', 'it']:
            assert 'arguments' in value[sub_key]
            assert 'defaults' in value[sub_key]
            assert 'reference_class' in value[sub_key]


def test_n2s_fgr_implementations_description():
    """Test that implementations_description returns descriptions."""
    n2s = Noise2SelfFGR()
    descriptions = n2s.implementations_description
    impls = n2s.implementations
    assert len(descriptions) == len(impls)
    for desc in descriptions:
        assert isinstance(desc, str)
        assert len(desc) > 0


def test_n2s_fgr_get_regressor_with_variant():
    """Test that get_regressor returns correct type for each variant."""
    from aydin.regression.cb import CBRegressor
    from aydin.regression.linear import LinearRegressor

    variants = [
        ('cb', CBRegressor),
        ('linear', LinearRegressor),
    ]

    try:
        from aydin.regression.lgbm import LGBMRegressor

        variants.append(('lgbm', LGBMRegressor))
    except (ImportError, OSError):
        pass  # LightGBM unavailable, skip lgbm variant

    for variant, expected_class in variants:
        n2s = Noise2SelfFGR(variant=variant)
        regressor = n2s.get_regressor()
        assert isinstance(regressor, expected_class)


def test_n2s_fgr_get_generator_default():
    """Test that get_generator returns StandardFeatureGenerator by default."""
    from aydin.features.standard_features import StandardFeatureGenerator

    n2s = Noise2SelfFGR()
    generator = n2s.get_generator()
    assert isinstance(generator, StandardFeatureGenerator)


def test_n2s_fgr_train_denoise_linear():
    """Test full train+denoise cycle with linear variant (fastest)."""
    image, noisy = _make_noisy_image()
    transforms = [{"class": RangeTransform, "kwargs": {}}]

    n2s = Noise2SelfFGR(variant='linear', it_transforms=transforms)
    n2s.train(noisy)
    denoised = n2s.denoise(noisy)

    denoised = denoised.clip(0, 1)
    assert denoised.shape == image.shape
    assert denoised.dtype == noisy.dtype


def test_n2s_fgr_repr():
    """Test string representation of N2S FGR denoiser."""
    n2s = Noise2SelfFGR(variant='cb')
    repr_str = repr(n2s)
    assert 'Noise2SelfFGR' in repr_str


def test_n2s_fgr_default_transforms():
    """Test that default transforms include Range, Padding, and VST."""
    n2s = Noise2SelfFGR()
    assert len(n2s.it_transforms) == 3


@pytest.mark.heavy
def test_n2s_fgr_train_denoise_lgbm():
    """Test full train+denoise cycle with lgbm variant."""
    image, noisy = _make_noisy_image()
    transforms = [{"class": RangeTransform, "kwargs": {}}]

    n2s = Noise2SelfFGR(variant='lgbm', it_transforms=transforms)
    n2s.train(noisy)
    denoised = n2s.denoise(noisy)

    denoised = denoised.clip(0, 1)
    assert denoised.shape == image.shape
    assert psnr(image, denoised) > psnr(image, noisy.clip(0, 1))


@pytest.mark.heavy
def test_n2s_fgr_convenience_function():
    """Test the module-level noise2self_fgr() convenience function."""
    image, noisy = _make_noisy_image()
    denoised = noise2self_fgr(noisy, variant='linear')

    denoised = denoised.clip(0, 1)
    assert denoised.shape == image.shape
