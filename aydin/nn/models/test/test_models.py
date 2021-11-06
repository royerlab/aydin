import numpy as np

from aydin.nn.models.jinet import JINetModel
from aydin.nn.models.unet import UNetModel


def test_supervised_2D():
    input_array = np.zeros((1, 64, 64, 1), dtype=np.float32)
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


def test_shiftconv_2D():
    input_array = np.zeros((1, 64, 64, 1), dtype=np.float32)
    model2d = UNetModel(
        (64, 64, 1),
        nb_unet_levels=2,
        shiftconv=True,
        supervised=False,
        spacetime_ndim=2,
    )
    result = model2d.predict(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_masking_2D():
    input_array = np.zeros((1, 64, 64, 1), dtype=np.float32)
    model2d = UNetModel(
        (64, 64, 1),
        nb_unet_levels=2,
        shiftconv=False,
        supervised=False,
        spacetime_ndim=2,
    )
    result = model2d.predict([input_array, input_array])
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_jinet_2D():
    input_array = np.zeros((1, 64, 64, 1), dtype=np.float32)
    model2d = JINetModel((64, 64, 1), spacetime_ndim=2)
    result = model2d.predict([input_array])
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_supervised_3D():
    input_array = np.zeros((1, 64, 64, 64, 1), dtype=np.float32)
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


def test_shiftconv_3D():
    input_array = np.zeros((1, 64, 64, 64, 1), dtype=np.float32)
    model3d = UNetModel(
        (64, 64, 64, 1),
        nb_unet_levels=2,
        shiftconv=True,
        supervised=False,
        spacetime_ndim=3,
    )
    result = model3d.predict(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_masking_3D():
    input_array = np.zeros((1, 64, 64, 64, 1), dtype=np.float32)
    model3d = UNetModel(
        (64, 64, 64, 1),
        nb_unet_levels=2,
        shiftconv=False,
        supervised=False,
        spacetime_ndim=3,
    )
    result = model3d.predict([input_array, input_array])
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_various_masking_3D():
    for i in [0, 4]:
        input_array = np.zeros((1, 21 + i, 64, 64, 1), dtype=np.float32)
        print(f'input shape: {input_array.shape}')
        model3d = UNetModel(
            input_array.shape[1:],
            nb_unet_levels=4,
            shiftconv=False,
            supervised=False,
            spacetime_ndim=3,
        )
        result = model3d.predict([input_array, input_array])
        assert result.shape == input_array.shape
        assert result.dtype == input_array.dtype


def test_thin_masking_3D():
    for i in range(3):
        input_array = np.zeros((1, 2 + i, 64, 64, 1), dtype=np.float32)
        print(f'input shape: {input_array.shape}')
        model3d = UNetModel(
            input_array.shape[1:],
            nb_unet_levels=4,
            shiftconv=False,
            supervised=False,
            spacetime_ndim=3,
        )
        result = model3d.predict([input_array, input_array])
        assert result.shape == input_array.shape
        assert result.dtype == input_array.dtype
