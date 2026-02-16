"""Tests for ImageTranslatorBase core functionality."""

import math

import numpy
import pytest

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import camera, normalise
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.attenuation import AttenuationTransform
from aydin.it.transforms.deskew import DeskewTransform
from aydin.it.transforms.fixedpattern import FixedPatternTransform
from aydin.it.transforms.histogram import HistogramEqualisationTransform
from aydin.it.transforms.motion import MotionStabilisationTransform
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.it.transforms.salt_pepper import SaltPepperTransform
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform
from aydin.regression.cb import CBRegressor


def test_get_tilling_strategy_and_margins():
    """Test tiling strategy and margin computation for different memory limits."""
    regressor = CBRegressor()
    generator = StandardFeatureGenerator()

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    image = normalise(camera().astype(numpy.float32))[numpy.newaxis, numpy.newaxis, ...]

    tilling_strategy, margins = it._get_tilling_strategy_and_margins(
        image, max_voxels_per_tile=128 * 128, min_margin=8, max_margin=math.inf
    )

    assert tilling_strategy == (1, 1, 4, 4)
    assert margins == (0, 0, 8, 8)

    it = ImageTranslatorFGR(
        feature_generator=generator,
        regressor=regressor,
        max_memory_usage_ratio=0.0000001,
    )

    image = normalise(camera().astype(numpy.float32))[numpy.newaxis, numpy.newaxis, ...]

    tilling_strategy, margins = it._get_tilling_strategy_and_margins(
        image, max_voxels_per_tile=128 * 128, min_margin=8, max_margin=math.inf
    )

    assert tilling_strategy[2] > 4 and tilling_strategy[3] > 4
    assert margins == (0, 0, 8, 8)


@pytest.mark.parametrize(
    "batch_axes, channel_axes, ndim, expected_batch_axes, expected_channel_axes",
    [
        (
            [False, True, False, False],
            [True, False, False, False],
            4,
            [False, True, False, False],
            [True, False, False, False],
        ),
        ([1], [0], 4, [False, True, False, False], [True, False, False, False]),
        (
            [False, False, False, False],
            [True, False, False, False],
            4,
            [False, False, False, False],
            [True, False, False, False],
        ),
        ([], [0], 4, [False, False, False, False], [True, False, False, False]),
        (
            [False, False, False, False, False],
            [True, False, False, False, False],
            5,
            [False, False, False, False, False],
            [True, False, False, False, False],
        ),
        (
            [],
            [0],
            5,
            [False, False, False, False, False],
            [True, False, False, False, False],
        ),
    ],
)
def test_parse_axes_args(
    batch_axes, channel_axes, ndim, expected_batch_axes, expected_channel_axes
):
    """Test that parse_axes_args correctly normalizes axis specifications."""
    it = ImageTranslatorFGR()

    result_batch_arg, result_chan_arg = it.parse_axes_args(
        batch_axes, channel_axes, ndim
    )
    assert result_batch_arg == expected_batch_axes
    assert result_chan_arg == expected_channel_axes


@pytest.mark.parametrize(
    "batch_axes, channel_axes, ndim",
    [
        ([True, True, True, True], [False, False, False, False], 4),
        ([0, 1, 2, 3], [], 4),
        ([False, False, False, False], [True, True, True, True], 4),
        ([], [0, 1, 2, 3], 4),
        ([True, False, False, False], [True, False, False, False], 4),
        ([0], [0], 4),
        ([False, False, False, False, False], [False, False, False, False, False], 5),
        ([], [], 5),
    ],
)
def test_parse_axes_args_raises_exception(batch_axes, channel_axes, ndim):
    """Test that parse_axes_args raises an exception for invalid axis configurations."""
    it = ImageTranslatorFGR()

    with pytest.raises(Exception):
        it.parse_axes_args(batch_axes, channel_axes, ndim)


def test_transform_sorting():
    """Test that transforms are sorted by priority after being added."""
    it = ImageTranslatorFGR()

    it.add_transform(AttenuationTransform())
    it.add_transform(RangeTransform())
    it.add_transform(SaltPepperTransform())
    it.add_transform(DeskewTransform())
    it.add_transform(FixedPatternTransform())
    it.add_transform(HistogramEqualisationTransform())
    it.add_transform(MotionStabilisationTransform())
    it.add_transform(PaddingTransform())
    it.add_transform(VarianceStabilisationTransform())

    for i, t1 in enumerate(it.transforms_list):
        for j, t2 in enumerate(it.transforms_list):
            if i < j:
                assert t1.priority < t2.priority


def test_blindspot_shorthand_notation():
    """Test parsing of blind-spot shorthand notation strings."""
    it = ImageTranslatorFGR()

    def _get_blindspot_parsed(blind_spots):
        """Parse blind-spot shorthand notation for 2D spatial dims."""
        return it._parse_blind_spot_shorthand_notation(blind_spots, st_ndim=2)

    blind_spots = _get_blindspot_parsed('0#2')
    print(blind_spots)
    assert blind_spots == [(0, 0), (2, 0), (-2, 0), (-1, 0), (1, 0)]

    blind_spots = _get_blindspot_parsed('y#2')
    print(blind_spots)
    assert blind_spots == [(0, 0), (2, 0), (-2, 0), (-1, 0), (1, 0)]

    blind_spots = _get_blindspot_parsed('1#2')
    print(blind_spots)
    assert blind_spots == [(0, 1), (0, 0), (0, -1), (0, 2), (0, -2)]

    blind_spots = _get_blindspot_parsed('x#2')
    print(blind_spots)
    assert blind_spots == [(0, 1), (0, 0), (0, -1), (0, 2), (0, -2)]

    blind_spots = _get_blindspot_parsed('x#0')
    print(blind_spots)
    assert blind_spots == [(0, 0)]

    blind_spots = _get_blindspot_parsed('x#5')
    print(blind_spots)
    assert blind_spots == [
        (0, 1),
        (0, -3),
        (0, 4),
        (0, 0),
        (0, 3),
        (0, -1),
        (0, -4),
        (0, 2),
        (0, 5),
        (0, -5),
        (0, -2),
    ]


def test_tiling_strategy_3d():
    """Test tiling strategy computation on a 3D image."""
    from skimage.data import binary_blobs

    regressor = CBRegressor()
    generator = StandardFeatureGenerator()

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    image = binary_blobs(length=64, n_dim=3, rng=1).astype(numpy.float32)
    image = image[numpy.newaxis, numpy.newaxis, ...]

    tilling_strategy, margins = it._get_tilling_strategy_and_margins(
        image, max_voxels_per_tile=32 * 32 * 32, min_margin=4, max_margin=math.inf
    )

    # Batch and channel axes should not be tiled
    assert tilling_strategy[0] == 1
    assert tilling_strategy[1] == 1
    # Spatial axes may need tiling
    assert len(tilling_strategy) == 5
    assert all(m >= 0 for m in margins)


def test_tiling_strategy_memory_constrained():
    """Test tiling with very low memory constraint forces more tiles."""
    regressor = CBRegressor()
    generator = StandardFeatureGenerator()

    it = ImageTranslatorFGR(
        feature_generator=generator,
        regressor=regressor,
        max_memory_usage_ratio=0.00000001,
    )

    image = normalise(camera().astype(numpy.float32))[numpy.newaxis, numpy.newaxis, ...]

    tilling_strategy, margins = it._get_tilling_strategy_and_margins(
        image, max_voxels_per_tile=64 * 64, min_margin=4, max_margin=math.inf
    )

    # With extreme memory constraint, more tiles should be needed
    assert tilling_strategy[2] > 1 or tilling_strategy[3] > 1


def test_parse_axes_args_5d():
    """Test axis parsing for 5-dimensional images with integer indices."""
    it = ImageTranslatorFGR()

    # Batch at 0, channel at 1
    batch_axes, channel_axes = it.parse_axes_args([0], [1], 5)
    assert batch_axes == [True, False, False, False, False]
    assert channel_axes == [False, True, False, False, False]


def test_transform_count():
    """Test that transforms list length matches number added."""
    it = ImageTranslatorFGR()

    it.add_transform(RangeTransform())
    it.add_transform(PaddingTransform())

    assert len(it.transforms_list) == 2


def test_default_no_transforms():
    """Test that new ImageTranslatorFGR starts with empty transforms list."""
    it = ImageTranslatorFGR()
    assert len(it.transforms_list) == 0


def test_clear_transforms():
    """Test that clear_transforms() empties the list."""
    it = ImageTranslatorFGR()
    it.add_transform(RangeTransform())
    it.add_transform(PaddingTransform())
    assert len(it.transforms_list) == 2
    it.clear_transforms()
    assert len(it.transforms_list) == 0


def test_add_transform_no_sort():
    """Test that sort=False preserves insertion order."""
    it = ImageTranslatorFGR()
    # PaddingTransform has higher priority than RangeTransform
    it.add_transform(PaddingTransform(), sort=False)
    it.add_transform(RangeTransform(), sort=False)
    # Should be in insertion order, not priority order
    assert isinstance(it.transforms_list[0], PaddingTransform)
    assert isinstance(it.transforms_list[1], RangeTransform)


def test_transform_preprocess_postprocess_roundtrip():
    """Test that preprocess+postprocess roundtrips through RangeTransform."""
    it = ImageTranslatorFGR()
    it.add_transform(RangeTransform())

    image = numpy.random.RandomState(0).rand(64, 64).astype(numpy.float32) * 100
    original = image.copy()

    preprocessed = it.transform_preprocess_image(image)
    # After range normalization, values should be in [0, 1]
    assert preprocessed.min() >= -0.1
    assert preprocessed.max() <= 1.1

    postprocessed = it.transform_postprocess_image(preprocessed)
    numpy.testing.assert_allclose(postprocessed, original, atol=1e-3)


def test_estimate_memory_needed_and_available():
    """Test memory estimation returns valid tuple (base implementation)."""
    from aydin.it.base import ImageTranslatorBase

    # Use base implementation directly via super()
    it = ImageTranslatorFGR()
    image = numpy.zeros((64, 64), dtype=numpy.float32)
    # Call the base class method explicitly (it returns (0, total_mem))
    needed, available = ImageTranslatorBase._estimate_memory_needed_and_available(
        it, image
    )
    assert needed >= 0
    assert available > 0


def test_stop_training_no_raise():
    """Test that stop_training() doesn't raise on base class."""
    it = ImageTranslatorFGR()
    it.stop_training()  # Should not raise


@pytest.mark.heavy
def test_save_load_roundtrip(tmp_path):
    """Test that save/load roundtrip preserves translator state."""
    from aydin.it.base import ImageTranslatorBase

    generator = StandardFeatureGenerator()
    regressor = CBRegressor()
    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    # Train on a small image so internal state is initialized
    image = normalise(camera()[:64, :64].astype(numpy.float32))
    it.train(image)

    save_path = str(tmp_path / 'test_translator')
    it.save(save_path)

    loaded = ImageTranslatorBase.load(save_path)
    assert isinstance(loaded, ImageTranslatorFGR)


def test_base_getstate_excludes_normalisers():
    """Base __getstate__ should exclude input_normaliser and target_normaliser."""
    from aydin.it.base import ImageTranslatorBase

    it = ImageTranslatorFGR()
    # Simulate normalisers being set during training
    it.input_normaliser = object()
    it.target_normaliser = object()

    # Call the base class __getstate__ directly
    state = ImageTranslatorBase.__getstate__(it)
    assert 'input_normaliser' not in state
    assert 'target_normaliser' not in state


def test_fgr_getstate_excludes_internals():
    """FGR __getstate__ should exclude feature_generator and regressor."""
    it = ImageTranslatorFGR()
    state = it.__getstate__()
    assert isinstance(state, dict)
    assert 'feature_generator' not in state
    assert 'regressor' not in state
    assert 'transforms_list' in state


def test_train_raises_on_mismatched_batch_axes():
    """train() should raise ArrayShapeDoesNotMatchError for wrong batch_axes length."""
    from aydin.it.exceptions.base import ArrayShapeDoesNotMatchError

    it = ImageTranslatorFGR()
    image = numpy.random.rand(64, 64).astype(numpy.float32)

    with pytest.raises(ArrayShapeDoesNotMatchError):
        # batch_axes has 3 elements but image is 2D
        it.train(image, batch_axes=[False, False, False])


def test_train_raises_on_shape_mismatch():
    """train() should raise when input and target spatial shapes differ."""
    from aydin.it.exceptions.base import ArrayShapeDoesNotMatchError

    it = ImageTranslatorFGR()
    input_image = numpy.random.rand(64, 64).astype(numpy.float32)
    target_image = numpy.random.rand(32, 32).astype(numpy.float32)

    with pytest.raises(ArrayShapeDoesNotMatchError):
        it.train(input_image, target_image=target_image)


def test_transform_preprocess_postprocess_multiple_transforms():
    """Preprocess+postprocess should roundtrip with multiple transforms."""
    it = ImageTranslatorFGR()
    it.add_transform(RangeTransform())
    it.add_transform(PaddingTransform())

    image = numpy.random.RandomState(0).rand(64, 64).astype(numpy.float32) * 100
    original = image.copy()

    preprocessed = it.transform_preprocess_image(image)
    postprocessed = it.transform_postprocess_image(preprocessed)

    # After padding+range and then unpadding+unrange, shape should be restored
    assert postprocessed.shape == original.shape


def test_translate_raises_on_mismatched_batch_axes():
    """translate() should raise ArrayShapeDoesNotMatchError for wrong batch_axes."""
    from aydin.it.exceptions.base import ArrayShapeDoesNotMatchError

    generator = StandardFeatureGenerator()
    regressor = CBRegressor()
    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    image = numpy.random.rand(64, 64).astype(numpy.float32)

    with pytest.raises(ArrayShapeDoesNotMatchError):
        it.translate(image, batch_axes=[False, False, False])
