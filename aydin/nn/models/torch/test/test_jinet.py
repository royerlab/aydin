import numpy
import torch

from aydin.io.datasets import add_noise, normalise, camera
from aydin.nn.models.torch.torch_jinet import JINetModel, n2t_jinet_train_loop
from aydin.nn.pytorch.it_ptcnn import to_numpy


def test_forward_2D():
    input_array = torch.zeros((1, 1, 64, 64))
    model2d = JINetModel(spacetime_ndim=2)
    result = model2d(input_array)

    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_forward_3D():
    input_array = torch.zeros((1, 1, 128, 128, 128))
    model3d = JINetModel(spacetime_ndim=3)
    result = model3d(input_array)
    assert result.shape == input_array.shape
    assert result.dtype == input_array.dtype


def test_supervised_2D_n2t():
    visualize = False
    lizard_image = normalise(camera()[:256, :256])
    lizard_image = numpy.expand_dims(lizard_image, axis=0)
    lizard_image = numpy.expand_dims(lizard_image, axis=0)

    input_image = add_noise(lizard_image)

    input_image = torch.tensor(input_image)
    lizard_image = torch.tensor(lizard_image)

    model = JINetModel(spacetime_ndim=2)

    n2t_jinet_train_loop(input_image, lizard_image, model)

    denoised = model(input_image)

    if visualize:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(to_numpy(lizard_image), name="groundtruth")
        viewer.add_image(to_numpy(input_image), name="noisy")
        viewer.add_image(to_numpy(denoised), name="denoised")

        napari.run()

    # assert result.shape == input_image.shape
    # assert result.dtype == input_image.dtype
