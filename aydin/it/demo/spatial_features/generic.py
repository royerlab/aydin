"""Demo comparing denoising with and without spatial features on generic images.

Trains two ``ImageTranslatorFGR`` models -- one with spatial features
enabled and one without -- and compares their PSNR/SSIM results.
Also tests translational invariance by denoising a flipped image.
"""

from functools import partial

# flake8: noqa
import napari
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity

ssim = partial(structural_similarity, data_range=1.0)

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import add_noise, newyork, normalise
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor


def demo(image, max_level=3):
    """Run denoising with and without spatial features and compare results.

    Parameters
    ----------
    image : numpy.ndarray
        Clean 2D reference image.
    max_level : int, optional
        Maximum feature pyramid level for the feature generator.
    """
    image = normalise(image.astype(np.float32))
    noisy = add_noise(image)

    # run wsf:
    generator = StandardFeatureGenerator(
        include_spatial_features=True, max_level=max_level
    )
    regressor = CBRegressor(patience=30)
    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)
    it.train(noisy, noisy)
    denoised_wsf = it.translate(noisy)

    # run wsf on flipped image
    flipped_img = np.flipud(noisy)
    denoised_wsf_flipped = np.flipud(it.translate(flipped_img))

    # run:
    generator = StandardFeatureGenerator(max_level=max_level)
    regressor = CBRegressor(patience=30)
    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)
    it.train(noisy, noisy)
    denoised = it.translate(noisy)

    # stats
    image_clipped = np.clip(image, 0, 1)
    noisy_clipped = np.clip(noisy, 0, 1)
    psnr_noisy = psnr(image_clipped, noisy_clipped)
    ssim_noisy = ssim(image_clipped, noisy_clipped)

    denoised_wsf_clipped = np.clip(denoised_wsf, 0, 1)
    psnr_denoised_wsf_clipped = psnr(image_clipped, denoised_wsf_clipped)
    ssim_denoised_wsf_clipped = ssim(image_clipped, denoised_wsf_clipped)

    denoised_clipped = np.clip(denoised, 0, 1)
    psnr_denoised_clipped = psnr(image_clipped, denoised_clipped)
    ssim_denoised_clipped = ssim(image_clipped, denoised_clipped)

    denoised_wsf_flipped_clipped = np.clip(denoised_wsf_flipped, 0, 1)
    psnr_denoised_wsf_flipped_clipped = psnr(
        image_clipped, denoised_wsf_flipped_clipped
    )
    ssim_denoised_wsf_flipped_clipped = ssim(
        image_clipped, denoised_wsf_flipped_clipped
    )

    print("noisy   :", psnr_noisy, ssim_noisy)
    print("denoised_wsf:", psnr_denoised_wsf_clipped, ssim_denoised_wsf_clipped)
    print("denoised:", psnr_denoised_clipped, ssim_denoised_clipped)
    print(
        "denoised_wsf_flipped:",
        psnr_denoised_wsf_flipped_clipped,
        ssim_denoised_wsf_flipped_clipped,
    )

    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(noisy, name='noisy')
    viewer.add_image(denoised_wsf, name='denoised_with_sf')
    viewer.add_image(denoised, name='denoised_without_sf')
    viewer.add_image(denoised_wsf_flipped, name='denoised_with_sf_flipped')

    # imsave("ss5_wsf.tif", denoised_wsf)
    # imsave("ss5.tif", denoised)
    napari.run()


if __name__ == '__main__':
    np.random.seed(0)
    # image = camera()
    # demo(image)
    image = newyork()
    demo(image)
