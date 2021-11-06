import numpy

from aydin.analysis.camera_simulation import simulate_camera_image
from aydin.io.datasets import characters
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform


def demo_vst():

    image = characters()
    image = image.astype(numpy.float32) * 0.1
    noisy = simulate_camera_image(image)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(image, name='noisy')

        for mode in [
            'yeo-johnson',
            'box-cox',
            'quantile',
            'anscomb',
            'log',
            'sqrt',
            'identity',
        ]:
            print(f"testing mode: {mode}")
            vst = VarianceStabilisationTransform(mode=mode)

            preprocessed = vst.preprocess(noisy)
            postprocessed = vst.postprocess(preprocessed)

            viewer.add_image(preprocessed, name='preprocessed_' + mode)
            viewer.add_image(postprocessed, name='postprocessed_' + mode)


demo_vst()
