from aydin.analysis.test.test_dimension_analysis import (
    test_dimension_analysis_maitre,
    test_dimension_analysis_hela,
    test_dimension_analysis_cognet,
    test_dimension_analysis_huang,
    test_dimension_analysis_royer,
)
from aydin.util.log.log import Log


def demo_dimension_analysis():

    Log.enable_output = True
    Log.set_log_max_depth(8)

    test_dimension_analysis_royer()
    test_dimension_analysis_huang()
    test_dimension_analysis_maitre()
    test_dimension_analysis_hela()
    test_dimension_analysis_cognet()


demo_dimension_analysis()
