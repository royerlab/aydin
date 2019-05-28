import numpy as np
import scipy
from napari.util import app_context
from scipy import ndimage as ndi
from scipy.signal import correlate2d
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.restoration import estimate_sigma
from tifffile import imread


def compute_features(image, width=5, scales=[1, 3, 9, 11], exclude_center=True):
    filters = []

    for scale in scales:
        base = np.zeros((width, width))
        for i in range(width * width):
            phases = np.unravel_index(i, (width, width))
            if exclude_center and phases[0] == phases[1] and phases[0] == width // 2:
                continue
            base = np.zeros((width, width))
            base[phases] = 1
            filters.append(base.repeat(scale, 0).repeat(scale, 1))

    features = [ndi.convolve(image, f, mode='mirror') for f in filters]
    features = np.stack(features, axis=-1)

    return features


def lgbm_map(input_image, target_image, rounds=1, viewer=None):
    denoised_images = []

    self_supervised = input_image is target_image

    features = compute_features(input_image, width=3, scales=[1, 3, 5, 7], exclude_center=self_supervised)

    x = features.reshape(-1, features.shape[-1])
    y = target_image.reshape(-1)

    nb_entries = y.shape[0]
    y_test = y[0:nb_entries // 10]
    y_train = y[nb_entries // 10:]

    from lightgbm import LGBMRegressor

    for round in range(rounds):
        print("Round: %d/%d" % (round + 1, rounds))
        gbm = LGBMRegressor(  # max_depth=16,
            num_leaves=63,
            learning_rate=0.05,
            n_estimators=256)

        nb_features = x.shape[-1]

        print("Number of entries: %d features: %d" % (nb_entries, nb_features))

        x_test = x[0:nb_entries // 10]
        x_train = x[nb_entries // 10:]

        gbm.fit(x_train, y_train,
                eval_metric='l1',
                eval_set=[(x_test, y_test)],
                early_stopping_rounds=5)

        z = gbm.predict(x, num_iteration=gbm.best_iteration_)
        denoised_image = z.reshape(image.shape).clip(0, 1)
        if viewer:
            viewer.add_image(denoised_image, name='lgbm%d' % round)
        denoised_images.append(denoised_image)
        print(psnr(denoised_image, image), ssim(denoised_image, image))

        if rounds > 1 and round < rounds - 1:
            new_features = compute_features(denoised_image, width=1, scales=[1], exclude_center=False)
            new_x = new_features.reshape(-1, new_features.shape[-1])
            x = np.concatenate([new_x, x], axis=1)

    return denoised_images


def autocorr2d(x):
    result = correlate2d(x, x)
    return result  # [result.shape[0] // 2:, result.shape[1] // 2:]


from napari import ViewerApp

with app_context():
    viewer = ViewerApp()

    image = imread('../../data/tribolium/train/GT/montage.tif').astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))
    viewer.add_image(image, name='image')

    noisy = imread('../../data/tribolium/train/low/montage.tif').astype(np.float32)
    noisy = rescale_intensity(noisy, in_range='image', out_range=(0, 1))

    estimate_sigma_noisy_ = sigma = estimate_sigma(noisy)
    print("Noisy:", psnr(noisy, image), ssim(noisy, image))
    viewer.add_image(noisy, name='noisy')

    denoised_median = scipy.signal.medfilt(noisy, 5)
    print("Median:", psnr(denoised_median, image), ssim(denoised_median, image))
    viewer.add_image(denoised_median, name='median')

    # denoised_nlm = denoise_nl_means(noisy, estimate_sigma_noisy_, multichannel=False)
    # print("NLM:", psnr(denoised_nlm, image), ssim(denoised_nlm, image))
    # viewer.add_image(denoised_nlm, name='nlm')
    #
    # denoised_tvsb = denoise_tv_bregman(noisy, weight=0.1)
    # print("TVSB:", psnr(denoised_tvsb, image), ssim(denoised_tvsb, image))
    # viewer.add_image(denoised_tvsb, name='tvsb')
    #
    # denoised_wav = denoise_wavelet(noisy, sigma=estimate_sigma_noisy_)
    # print("Wavelet:", psnr(denoised_wav, image), ssim(denoised_wav, image))
    # viewer.add_image(denoised_wav, name='wavelet')
    #
    denoised_gbm_n2t = lgbm_map(noisy, image, rounds=1, viewer=viewer)
    print("LGBM N2T:", psnr(denoised_gbm_n2t[-1], image), ssim(denoised_gbm_n2t[-1], image))
    viewer.add_image(denoised_gbm_n2t[-1], name='LGBM_n2t')

    denoised_gbm_n2s = lgbm_map(noisy, noisy, rounds=1, viewer=viewer)
    print("LGBM N2S:", psnr(denoised_gbm_n2s[-1], image), ssim(denoised_gbm_n2s[-1], image))
    viewer.add_image(denoised_gbm_n2s[-1], name='LGBM_n2s')
