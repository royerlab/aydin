import numpy

from aydin.nn.models.torch_unet import UNetModel


def test_supervised_2D():
    input_array = numpy.zeros((1, 64, 64, 1), dtype=numpy.float32)
    model2d = UNetModel(
        (64, 64, 1),
        nb_unet_levels=2,
        shiftconv=False,
        supervised=True,
        spacetime_ndim=2,
    )
    result = model2d.predict(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_supervised_3D():
    input_array = numpy.zeros((1, 64, 64, 64, 1), dtype=numpy.float32)
    model3d = UNetModel(
        (64, 64, 64, 1),
        nb_unet_levels=2,
        shiftconv=False,
        supervised=True,
        spacetime_ndim=3,
    )
    result = model3d.predict(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype
