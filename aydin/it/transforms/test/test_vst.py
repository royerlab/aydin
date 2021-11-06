import numpy

from aydin.analysis.camera_simulation import simulate_camera_image
from aydin.io.datasets import characters
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform


def test_vst():
    image = characters()
    image = image.astype(numpy.float32) * 0.1
    noisy = simulate_camera_image(image)

    for mode in [
        'box-cox',
        'yeo-johnson',
        'quantile',
        'anscomb',
        'log',
        'sqrt',
        'identity',
    ]:
        print(f"testing mode: {mode}")
        vst = VarianceStabilisationTransform(mode=mode, leave_as_float=False)

        preprocessed = vst.preprocess(noisy)
        postprocessed = vst.postprocess(preprocessed)

        # import napari
        # with napari.gui_qt():
        #     viewer = napari.Viewer()
        #     viewer.add_image(image, name='image')
        #     viewer.add_image(noisy, name='noisy')
        #     viewer.add_image(preprocessed, name='preprocessed')
        #     viewer.add_image(postprocessed, name='postprocessed')

        error = numpy.abs(
            postprocessed.astype(numpy.float32) - noisy.astype(numpy.float32)
        ).mean()

        print(f"round-trip error: {error}")

        assert error < 0.33

        assert postprocessed.dtype == noisy.dtype
        assert postprocessed.shape == noisy.shape
