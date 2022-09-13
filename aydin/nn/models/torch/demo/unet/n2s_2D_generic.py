# flake8: noqa
import time
import numpy
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from aydin.io.datasets import (
    normalise,
    add_noise,
    camera,
    newyork,
    lizard,
    pollen,
    dots,
    characters,
)
from aydin.nn.models.torch.torch_unet import UNetModel, n2s_train
from aydin.nn.models.utils.n2s_dataset import N2SDataset
from aydin.util.log.log import Log


def demo(image, do_add_noise=True):
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

    model = UNetModel(
        nb_unet_levels=2,
        spacetime_ndim=2,
    )

    print("training starts")

    start = time.time()
    n2s_train(noisy, model, nb_epochs=128)
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
    # newyork_image = newyork()
    # demo(newyork_image, "newyork")
    # lizard_image = lizard()
    # demo(lizard_image, "lizard")
    # characters_image = characters()
    # demo(characters_image, "characters")

    camera_image = camera()
    demo(camera_image, "camera")
    # pollen_image = pollen()
    # demo(pollen_image, "pollen")
    # dots_image = dots()
    # demo(dots_image, "dots")
