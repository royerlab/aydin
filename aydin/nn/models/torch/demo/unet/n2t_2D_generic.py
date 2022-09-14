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
from aydin.nn.models.torch.torch_res_unet import ResidualUNetModel
from aydin.nn.models.torch.torch_unet import UNetModel, n2t_train
from aydin.nn.models.utils.grid_masked_dataset import GridMaskedDataset
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

    noisy = torch.tensor(noisy)
    image = torch.tensor(image)

    model = ResidualUNetModel(nb_unet_levels=2, supervised=True, spacetime_ndim=2)

    print("training starts")

    start = time.time()
    n2t_train(noisy, image, model)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

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
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)
    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised)
    print("noisy   :", psnr_noisy, ssim_noisy)
    print("denoised:", psnr_denoised, ssim_denoised)

    import napari

    viewer = napari.Viewer()  # no prior setup needed
    viewer.add_image(image, name='image')
    viewer.add_image(noisy, name='noisy')
    viewer.add_image(denoised, name='denoised')
    napari.run()


# NOT Working
# newyork_image = newyork()
# demo(newyork_image, "newyork")
# lizard_image = lizard()
# demo(lizard_image, "lizard")
# characters_image = characters()
# demo(characters_image, "characters")

# Working
# camera_image = camera()
# demo(camera_image, "camera")
# pollen_image = pollen()
# demo(pollen_image, "pollen")
dots_image = dots()
demo(dots_image, "dots")
