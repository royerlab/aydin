# flake8: noqa
from random import randint

import napari
import numpy as np
from skimage.draw import line
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import add_noise
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor


def generate_lines_image(
    image_size=1000, number_of_lines=1200, min_line_length=15, max_line_length=50
):
    img = np.zeros((image_size, image_size), dtype=np.float32)

    # draw line
    def add_line(img, r0, c0, r1, c1):
        rr, cc = line(r0, c0, r1, c1)
        img[rr, cc] = 1.0

    # add horizontal lines on top half
    for _ in range(number_of_lines // 2):
        line_length = randint(min_line_length, max_line_length)
        col = randint(0, 949 - line_length)
        row = randint(0, 500)
        add_line(img, row, col, row, col + line_length)

    # add vertical lines on bottom half
    for _ in range(number_of_lines // 2):
        line_length = randint(min_line_length, max_line_length)
        col = randint(0, 999)
        row = randint(500, 949 - line_length)
        add_line(img, row, col, row + line_length, col)

    return img


def demo(image):
    # imsave("ss5.tif", img)

    noisy = add_noise(image, intensity=5, variance=0.1, sap=0.2)

    # imsave("ss5_noisy.tif", img)

    # run wsf:
    generator = StandardFeatureGenerator(
        max_level=8,
        include_corner_features=True,
        include_scale_one=True,
        include_fine_features=True,
        include_spatial_features=True,
    )
    regressor = CBRegressor(patience=30)
    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)
    it.train(noisy, noisy)
    denoised_wsf = it.translate(noisy)

    # run wsf on flipped image
    flipped_img = np.flipud(noisy)
    denoised_wsf_flipped = np.flipud(it.translate(flipped_img))

    # run:
    generator = StandardFeatureGenerator()
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

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(denoised_wsf, name='denoised_wsf')
        viewer.add_image(denoised, name='denoised')
        viewer.add_image(denoised_wsf_flipped, name='denoised_wsf_flipped')

        # imsave("ss5_wsf.tif", denoised_wsf)
        # imsave("ss5.tif", denoised)


if __name__ == '__main__':
    np.random.seed(0)
    image = generate_lines_image()
    demo(image)
