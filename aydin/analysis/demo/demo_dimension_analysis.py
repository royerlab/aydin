# flake8: noqa
from aydin.analysis.test.test_dimension_analysis import (
    test_dimension_analysis,
    test_dimension_analysis_flybrain,
    test_dimension_analysis_hela,
)
from aydin.io.datasets import examples_single
from aydin.util.log.log import Log


def demo_dimension_analysis(display=False):

    Log.enable_output = True
    Log.set_log_max_depth(8)

    # Run parametrized test with different inputs
    test_dimension_analysis(examples_single.noisy_newyork.get_array(), [], [])
    test_dimension_analysis(examples_single.maitre_mouse.get_array(), [0, 1], [])
    test_dimension_analysis_flybrain(display=display)
    test_dimension_analysis_hela(display=display)


if __name__ == "__main__":
    demo_dimension_analysis()
