# flake8: noqa
import time
import numpy
import torch

from aydin.io.datasets import (
    normalise,
    add_noise,
    camera, newyork,
)
from aydin.nn.models.torch.torch_linear_scaling_unet import LinearScalingUNetModel
from aydin.nn.models.torch.torch_res_unet import ResidualUNetModel
from aydin.nn.models.torch.torch_unet import UNetModel, n2s_train
from aydin.util.log.log import Log


def demo(image, model_class, do_add_noise=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Log.enable_output = True
    Log.set_log_max_depth(8)

    image = normalise(image)
    image = numpy.expand_dims(image, axis=0)
    image = numpy.expand_dims(image, axis=0)
    noisy = add_noise(image) if do_add_noise else image
    print(noisy.shape)

    # noisy = torch.tensor(noisy)
    image = torch.tensor(image)

    model = model_class(
        nb_unet_levels=2,
        spacetime_ndim=2,
    )

    print("training starts")

    start = time.time()
    n2s_train(noisy, model, nb_epochs=256)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    noisy = torch.tensor(noisy)
    model.eval()
    model = model.cpu()
    print(f"noisy tensor shape: {noisy.shape}")
    # in case of batching we have to do this:
    start = time.time()
    denoised = model(noisy)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    noisy = noisy.detach().numpy()[0, 0, :, :]
    image = image.detach().numpy()[0, 0, :, :]
    denoised = denoised.detach().numpy()[0, 0, :, :]

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)
    # psnr_noisy = psnr(image, noisy)
    # ssim_noisy = ssim(image, noisy)
    # psnr_denoised = psnr(image, denoised)
    # ssim_denoised = ssim(image, denoised)
    # print("noisy   :", psnr_noisy, ssim_noisy)
    # print("denoised:", psnr_denoised, ssim_denoised)

    import napari

    viewer = napari.Viewer()  # no prior setup needed
    viewer.add_image(image, name='image')
    viewer.add_image(noisy, name='noisy')
    viewer.add_image(denoised, name='denoised')
    napari.run()


if __name__ == '__main__':
    image = newyork()[512:, :512]
    # image = lizard()
    # image = characters()
    # image = camera()
    # image = pollen()
    # image = dots()

    model_class = UNetModel
    # model_class = ResidualUNetModel
    # model_class = LinearScalingUNetModel

    demo(image, model_class)
