import numpy
from skimage.data import binary_blobs

from aydin.io.datasets import normalise
from aydin.it.transforms.attenuation import AttenuationTransform


def test_attenuation_correction_real():
    image = binary_blobs(length=128, seed=1, n_dim=3).astype(numpy.float32)
    image = normalise(image)

    ramp = numpy.linspace(0, 1, 128).astype(numpy.float32)

    attenuated = image * ramp

    ac = AttenuationTransform(axes=2)

    preprocessed = ac.preprocess(attenuated)
    postprocessed = ac.postprocess(preprocessed)

    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(image, name='image')
    #     viewer.add_image(attenuated, name='attenuated')
    #     viewer.add_image(preprocessed, name='preprocessed')
    #     viewer.add_image(postprocessed, name='postprocessed')

    assert image.shape == postprocessed.shape
    assert image.dtype == postprocessed.dtype
    assert numpy.abs(preprocessed - image).mean() < 0.005

    assert postprocessed.dtype == attenuated.dtype
    assert numpy.abs(postprocessed - attenuated).mean() < 1e-8
