import numpy

from aydin.io.datasets import normalise, pollen, add_noise
from aydin.it.transforms.deskew import DeskewTransform


def test_deskew_positive():
    array = numpy.random.rand(10, 256, 256)

    sd = DeskewTransform(delta=1)

    processed = sd.preprocess(array)
    postprocessed = sd.postprocess(processed)

    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(array, name='array')
    #     viewer.add_image(processed, name='processed')
    #     viewer.add_image(postprocessed, name='postprocessed')

    print(f"array.shape = {array.shape}")
    print(f"processed.shape = {processed.shape}")
    print(f"postprocessed.shape = {postprocessed.shape}")

    assert array.shape == postprocessed.shape
    assert array.dtype == postprocessed.dtype
    assert (numpy.abs(postprocessed - array) < 0.00001).all()


def test_deskew_negative():
    array = numpy.random.rand(10, 10, 10, 10)

    sd = DeskewTransform(delta=-3)

    ds_array = sd.preprocess(array)
    s_array = sd.postprocess(ds_array)

    assert (numpy.abs(s_array - array) < 0.00001).all()


def test_deskew_with_non_standard_axes():
    array = numpy.random.rand(10, 10, 10, 10)

    sd = DeskewTransform(delta=3, z_axis=1, skew_axis=0)

    ds_array = sd.preprocess(array)
    s_array = sd.postprocess(ds_array)

    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(array, name='array')
    #     viewer.add_image(ds_array, name='ds_array')
    #     viewer.add_image(s_array, name='s_array')

    assert (numpy.abs(s_array - array) < 0.00001).all()


def test_deskew_sim():
    shifts = tuple((-3 * i, 0) for i in range(100))

    image = normalise(pollen())[0:256, 0:256]
    array = numpy.stack(
        [add_noise(numpy.roll(image, shift=shift, axis=(0, 1))) for shift in shifts]
    )

    sd = DeskewTransform(delta=3)

    ds_array = sd.preprocess(array)
    s_array = sd.postprocess(ds_array)

    assert (array == s_array).all()
