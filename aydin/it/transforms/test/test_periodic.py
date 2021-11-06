import numpy

from aydin.io.datasets import normalise, camera, add_noise
from aydin.it.transforms.periodic import PeriodicNoiseSuppressionTransform


def test_high_pass():
    image = normalise(camera().astype(numpy.float32))

    freq = 96
    periodic_pattern = 0.3 * (
        1 + numpy.cos(numpy.linspace(0, freq * 2 * numpy.pi, num=image.shape[0]))
    )
    periodic_pattern = periodic_pattern[:, numpy.newaxis]
    image += periodic_pattern

    image = add_noise(image)

    pns = PeriodicNoiseSuppressionTransform(post_processing_is_inverse=True)

    preprocessed = pns.preprocess(image)
    postprocessed = pns.postprocess(preprocessed)

    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(image, name='image')
    #     viewer.add_image(preprocessed, name='preprocessed')
    #     viewer.add_image(postprocessed, name='postprocessed')

    assert image.shape == postprocessed.shape
    assert image.dtype == postprocessed.dtype

    assert postprocessed.dtype == image.dtype
    assert numpy.abs(postprocessed - image).mean() < (1e-2 / 2)
