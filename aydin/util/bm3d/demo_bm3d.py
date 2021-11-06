# flake8: noqa
"""
Grayscale BM3D denoising demo file, based on Y. Mäkinen, L. Azzari, A. Foi, 2019.
Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise.
In IEEE International Conference on Image Processing (ICIP), pp. 185-189
"""
import numpy
import numpy as np
import matplotlib.pyplot as plt
from bm3d import bm3d

from aydin.features.fast.fast_features import FastFeatureGenerator
from aydin.io.datasets import camera, cropped_newyork, newyork, characters, pollen
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor
from aydin.regression.lgbm import LGBMRegressor
from aydin.util.bm3d.experiment_funcs import (
    get_experiment_noise,
    get_psnr,
    get_cropped_psnr,
)


def main():

    # Load noise-free image
    y = np.array(characters()) / 255
    # Possible noise types to be generated 'gw', 'g1', 'g2', 'g3', 'g4', 'g1w',
    # 'g2w', 'g3w', 'g4w'.
    noise_type = 'gw'
    noise_var = 0.2  # Noise variance
    seed = 0  # seed for pseudorandom noise realization

    # Generate noise with given PSD
    noise, psd, kernel = get_experiment_noise(noise_type, noise_var, seed, y.shape)
    # N.B.: For the sake of simulating a more realistic acquisition scenario,
    # the generated noise is *not* circulant. Therefore there is a slight
    # discrepancy between PSD and the actual PSD computed from infinitely many
    # realizations of this noise with different seeds.

    # Generate noisy image corrupted by additive spatially correlated noise
    # with noise power spectrum PSD
    z = np.atleast_3d(y) + np.atleast_3d(noise)

    # Call BM3D With the default settings.
    y_est = bm3d(z, psd)

    # To include refiltering:
    # y_est = bm3d(z, psd, 'refilter')

    # For other settings, use BM3DProfile.
    # profile = BM3DProfile(); # equivalent to profile = BM3DProfile('np');
    # profile.gamma = 6;  # redefine value of gamma parameter
    # y_est = bm3d(z, psd, profile);

    # Note: For white noise, you may instead of the PSD
    # also pass a standard deviation
    # y_est = bm3d(z, sqrt(noise_var));

    # Ignore values outside range for display (or plt gives an error for multichannel input)
    y_est = np.minimum(np.maximum(y_est, 0), 1)
    z_rang = np.minimum(np.maximum(z, 0), 1)
    plt.title("y, z, y_est")
    plt.imshow(np.concatenate((y, np.squeeze(z_rang), y_est), axis=1), cmap='gray')
    plt.show()

    generator = FastFeatureGenerator(include_spatial_features=False)

    regressor = LGBMRegressor(patience=8)  # , gpu=True

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    z = z.squeeze()
    it.train(z, z)
    denoised = it.translate(z)

    y = numpy.clip(y, 0, 1)
    z = numpy.clip(z, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    psnr = get_psnr(y, y_est)
    ssim_value = ssim(y, y_est)
    print("BM3D PSNR:", psnr)
    print("BM3D SSIM:", ssim_value)

    psnr = get_psnr(y, denoised)
    ssim_value = ssim(y, denoised)
    print("AYDIN PSNR:", psnr)
    print("AYDIN SSIM:", ssim_value)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(y, name='y')
        viewer.add_image(z, name='z')
        viewer.add_image(y_est, name='y_est')
        viewer.add_image(denoised, name='denoised')


if __name__ == '__main__':
    main()
