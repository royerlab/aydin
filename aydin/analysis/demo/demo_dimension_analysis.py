# flake8: noqa
from aydin.analysis.test.test_dimension_analysis import (
    test_dimension_analysis_maitre,
    test_dimension_analysis_hela,
    test_dimension_analysis_cognet,
    test_dimension_analysis_royer,
    test_dimension_analysis_leonetti,
    test_dimension_analysis_myers,
    test_dimension_analysis_flybrain,
)
from aydin.util.log.log import Log


def demo_dimension_analysis(display=False):

    Log.enable_output = True
    Log.set_log_max_depth(8)

    # test_dimension_analysis_huang(display=display)
    test_dimension_analysis_leonetti(display=display)
    test_dimension_analysis_maitre(display=display)
    test_dimension_analysis_cognet(display=display)
    test_dimension_analysis_flybrain(display=display)
    test_dimension_analysis_myers(display=display)
    test_dimension_analysis_hela(display=display)
    test_dimension_analysis_royer(display=display)


if __name__ == "__main__":
    demo_dimension_analysis()
