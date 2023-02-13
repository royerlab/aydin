import math

import numpy
import pytest

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import normalise, camera
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.attenuation import AttenuationTransform
from aydin.it.transforms.salt_pepper import SaltPepperTransform
from aydin.it.transforms.deskew import DeskewTransform
from aydin.it.transforms.fixedpattern import FixedPatternTransform
from aydin.it.transforms.histogram import HistogramEqualisationTransform
from aydin.it.transforms.motion import MotionStabilisationTransform
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform
from aydin.regression.cb import CBRegressor


def test_get_tilling_strategy_and_margins():
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


def test_stop_training_classic():
    # TODO: implement test
    pass


def test_stop_training_cnn():
    # TODO: implement test
    pass


def test_train():
    # TODO: write a test that checks edge cases for train method inputs
    pass


@pytest.mark.parametrize(
    "batch_axes, chan_axes, ndim, expected_batch_axes, expected_chan_axes",
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
    batch_axes, chan_axes, ndim, expected_batch_axes, expected_chan_axes
):
    it = ImageTranslatorFGR()

    result_batch_arg, result_chan_arg = it.parse_axes_args(batch_axes, chan_axes, ndim)
    assert result_batch_arg == expected_batch_axes
    assert result_chan_arg == expected_chan_axes


@pytest.mark.parametrize(
    "batch_axes, chan_axes, ndim",
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
def test_parse_axes_args_raises_exception(batch_axes, chan_axes, ndim):
    it = ImageTranslatorFGR()

    with pytest.raises(Exception):
        it.parse_axes_args(batch_axes, chan_axes, ndim)


def test_transform_sorting():
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
    it = ImageTranslatorFGR()

    def _get_blindspot_parsed(blind_spots):
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
