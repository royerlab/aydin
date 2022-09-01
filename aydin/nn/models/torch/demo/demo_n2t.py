# flake8: noqa
import numpy
import torch

from aydin.io.datasets import add_noise, normalise, camera
from aydin.nn.models.torch.torch_linear_scaling_unet import LinearScalingUNetModel
from aydin.nn.models.torch.torch_res_unet import ResidualUNetModel
from aydin.nn.models.torch.torch_unet import n2t_unet_train_loop, UNetModel
from aydin.nn.pytorch.it_ptcnn import to_numpy


def demo_supervised_2D_n2t(model_class):
    visualize = True
    lizard_image = normalise(camera()[:256, :256])
    lizard_image = numpy.expand_dims(lizard_image, axis=0)
    lizard_image = numpy.expand_dims(lizard_image, axis=0)

    input_image = add_noise(lizard_image)

    input_image = torch.tensor(input_image)
    lizard_image = torch.tensor(lizard_image)

    model = model_class(nb_unet_levels=2, supervised=True, spacetime_ndim=2)

    n2t_unet_train_loop(input_image, lizard_image, model)

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


if __name__ == '__main__':
    model_class = UNetModel
    # model_class = ResidualUNetModel
    # model_class = LinearScalingUNetModel

    demo_supervised_2D_n2t(model_class)
