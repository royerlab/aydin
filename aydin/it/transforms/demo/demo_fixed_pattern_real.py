from aydin.io.datasets import examples_single
from aydin.it.transforms.fixedpattern import FixedPatternTransform
from aydin.util.log.log import Log


def demo_fixed_pattern_real():
    Log.override_test_exclusion = True

    image = examples_single.huang_fixed_pattern_noise.get_array()  # [:, 0:64, 0:64]

    bs = FixedPatternTransform()  # axes=[1, 2])

    pre_processed = bs.preprocess(image)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(pre_processed, name='pre_processed')


demo_fixed_pattern_real()
