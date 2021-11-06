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


def test_parse_axes_args_with_bool_lists():
    it = ImageTranslatorFGR()
    sample_batch_arg = [False, True, False, False]
    sample_chan_arg = [True, False, False, False]

    result_batch_arg, result_chan_arg = it.parse_axes_args(
        sample_batch_arg, sample_chan_arg, 4
    )
    assert result_batch_arg == sample_batch_arg
    assert result_chan_arg == sample_chan_arg


def test_parse_axes_args_with_index_lists():
    it = ImageTranslatorFGR()
    sample_batch_arg = [1]
    sample_chan_arg = [0]

    result_batch_arg, result_chan_arg = it.parse_axes_args(
        sample_batch_arg, sample_chan_arg, 4
    )
    assert result_batch_arg == [False, True, False, False]
    assert result_chan_arg == [True, False, False, False]


def test_parse_axes_args_with_bool_lists_3d_spacetime():
    it = ImageTranslatorFGR()
    sample_batch_arg = [False, False, False, False]
    sample_chan_arg = [True, False, False, False]

    result_batch_arg, result_chan_arg = it.parse_axes_args(
        sample_batch_arg, sample_chan_arg, 4
    )
    assert result_batch_arg == sample_batch_arg
    assert result_chan_arg == sample_chan_arg


def test_parse_axes_args_with_index_lists_3d_spacetime():
    it = ImageTranslatorFGR()
    sample_batch_arg = []
    sample_chan_arg = [0]

    result_batch_arg, result_chan_arg = it.parse_axes_args(
        sample_batch_arg, sample_chan_arg, 4
    )
    assert result_batch_arg == [False, False, False, False]
    assert result_chan_arg == [True, False, False, False]


def test_parse_axes_args_with_bool_lists_4d_spacetime():
    it = ImageTranslatorFGR()
    sample_batch_arg = [False, False, False, False, False]
    sample_chan_arg = [True, False, False, False, False]

    result_batch_arg, result_chan_arg = it.parse_axes_args(
        sample_batch_arg, sample_chan_arg, 5
    )
    assert result_batch_arg == sample_batch_arg
    assert result_chan_arg == sample_chan_arg


def test_parse_axes_args_with_index_lists_4d_spacetime():
    it = ImageTranslatorFGR()
    sample_batch_arg = []
    sample_chan_arg = [0]

    result_batch_arg, result_chan_arg = it.parse_axes_args(
        sample_batch_arg, sample_chan_arg, 5
    )
    assert result_batch_arg == [False, False, False, False, False]
    assert result_chan_arg == [True, False, False, False, False]


def test_parse_axes_args_with_all_batch():
    it = ImageTranslatorFGR()

    with pytest.raises(Exception):
        sample_batch_arg = [True, True, True, True]
        sample_chan_arg = [False, False, False, False]

        it.parse_axes_args(sample_batch_arg, sample_chan_arg, 4)

    with pytest.raises(Exception):
        sample_batch_arg = [0, 1, 2, 3]
        sample_chan_arg = []

        it.parse_axes_args(sample_batch_arg, sample_chan_arg, 4)


def test_parse_axes_args_with_all_chan():
    it = ImageTranslatorFGR()

    with pytest.raises(Exception):
        sample_batch_arg = [False, False, False, False]
        sample_chan_arg = [True, True, True, True]

        it.parse_axes_args(sample_batch_arg, sample_chan_arg, 4)

    with pytest.raises(Exception):
        sample_batch_arg = []
        sample_chan_arg = [0, 1, 2, 3]

        it.parse_axes_args(sample_batch_arg, sample_chan_arg, 4)


def test_parse_axes_args_with_colliding_batch_chan():
    it = ImageTranslatorFGR()

    with pytest.raises(Exception):
        sample_batch_arg = [True, False, False, False]
        sample_chan_arg = [True, False, False, False]

        it.parse_axes_args(sample_batch_arg, sample_chan_arg, 4)

    with pytest.raises(Exception):
        sample_batch_arg = [0]
        sample_chan_arg = [0]

        it.parse_axes_args(sample_batch_arg, sample_chan_arg, 4)


def test_parse_axes_args_with_5d_spacetime():
    it = ImageTranslatorFGR()

    with pytest.raises(Exception):
        sample_batch_arg = [False, False, False, False, False]
        sample_chan_arg = [False, False, False, False, False]

        it.parse_axes_args(sample_batch_arg, sample_chan_arg, 5)

    with pytest.raises(Exception):
        sample_batch_arg = []
        sample_chan_arg = []

        it.parse_axes_args(sample_batch_arg, sample_chan_arg, 5)


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
