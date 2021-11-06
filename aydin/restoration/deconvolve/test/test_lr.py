import numpy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.analysis.image_metrics import mutual_information, spectral_mutual_information
from aydin.io.datasets import characters, add_blur_2d, add_noise, normalise
from aydin.restoration.deconvolve.lr import LucyRichardson


def printscore(header, val1, val2, val3, val4):
    print(f"{header}: \t {val1:.4f} \t {val2:.4f} \t {val3:.4f} \t {val4:.4f}")


def test_run():
    # Prepare the input image
    image = normalise(characters().astype(numpy.float32))
    blurred_image, psf_kernel = add_blur_2d(image)
    noisy_and_blurred_image = add_noise(
        blurred_image, intensity=10000, variance=0.0001, sap=0.0000001
    )

    # Call the LucyRichardson restoration
    lr = LucyRichardson(
        psf_kernel=psf_kernel, max_num_iterations=20, backend='scipy-cupy'
    )

    lr.train(noisy_and_blurred_image, noisy_and_blurred_image)

    lr_deconvolved_image = lr.deconvolve(noisy_and_blurred_image)

    print("Below in order: PSNR, norm spectral mutual info, norm mutual info, SSIM: ")
    printscore(
        "n&b image",
        psnr(image, noisy_and_blurred_image),
        spectral_mutual_information(image, noisy_and_blurred_image),
        mutual_information(image, noisy_and_blurred_image),
        ssim(image, noisy_and_blurred_image),
    )

    printscore(
        "lr      ",
        psnr(image, lr_deconvolved_image),
        spectral_mutual_information(image, lr_deconvolved_image),
        mutual_information(image, lr_deconvolved_image),
        ssim(image, lr_deconvolved_image),
    )

    assert psnr(image, noisy_and_blurred_image) < psnr(image, lr_deconvolved_image)
    assert mutual_information(image, noisy_and_blurred_image) < mutual_information(
        image, lr_deconvolved_image
    )
    assert ssim(image, noisy_and_blurred_image) < ssim(image, lr_deconvolved_image)
    assert 19.0 < psnr(image, lr_deconvolved_image)
