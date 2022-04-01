from aydin.analysis.test.test_dimension_analysis import (
    test_dimension_analysis_maitre,
    test_dimension_analysis_hela,
    test_dimension_analysis_cognet,
    test_dimension_analysis_huang,
    test_dimension_analysis_royer,
    test_dimension_analysis_leonetti,
    test_dimension_analysis_myers,
    test_dimension_analysis_flybrain,
)
from aydin.util.log.log import Log


def demo_dimension_analysis():

    Log.enable_output = True
    Log.set_log_max_depth(8)

    test_dimension_analysis_hela(display=True)
    test_dimension_analysis_royer(display=True)
    test_dimension_analysis_cognet(display=True)
    test_dimension_analysis_flybrain(display=True)
    test_dimension_analysis_leonetti(display=True)
    test_dimension_analysis_myers(display=True)
    test_dimension_analysis_huang(display=True)
    test_dimension_analysis_maitre(display=True)


demo_dimension_analysis()
