from aydin.analysis.test.test_dimension_analysis import test_dimension_analysis
from aydin.util.log.log import Log


def demo_dimension_analysis():

    Log.enable_output = True
    Log.set_log_max_depth(8)
    test_dimension_analysis()


demo_dimension_analysis()
