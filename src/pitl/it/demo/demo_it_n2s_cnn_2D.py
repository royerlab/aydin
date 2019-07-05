import os
import time

import numpy as np
from matplotlib import pyplot as plt
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from src.pitl.features.multiscale_convolutions import MultiscaleConvolutionalFeatures
from src.pitl.pitl_classic import ImageTranslator
from src.pitl.regression.nn import CNNRegressor, Modeltype

"""
    Demo for self-supervised denoising using camera image with synthetic noise
"""

def demo_pitl_2D(noisy):
    scales = [1, 3, 7, 15, 31, 63, 127]
    widths = [3, 3, 3,  3,  3,  3,   3]

    start_time = time.time()
    regressor = CNNRegressor(mode=Modeltype.Convolutional,
                             learning_rate=0.001,
                             early_stopping_rounds=5)
    generator = MultiscaleConvolutionalFeatures(kernel_widths=widths,
                                                kernel_scales=scales,
                                                kernel_shapes=['l1'] * len(scales),
                                                exclude_center=True,
                                                )
    it = ImageTranslator(feature_generator=generator, regressor=regressor)
    denoised = it.train(noisy, noisy)

    results = [[psnr(noisy, image), ssim(noisy, image)]]
    results.append([psnr(denoised, image), ssim(denoised, image)])
    results.append(time.time() - start_time)

    print("noisy ", results[0])
    print("denoised ", results[1])
    print("time elapsed: ", results[2])
    return denoised, results

image = camera().astype(np.float32)
image = rescale_intensity(image, in_range='image', out_range=(0, 1))

intensity = 5
np.random.seed(0)
noisy = np.random.poisson(image * intensity) / intensity
noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
noisy = noisy.astype(np.float32)

denoised_cnn, results_cnn = demo_pitl_2D(noisy)

base_path = os.path.dirname(__file__)
savepath = os.path.join(base_path, 'output_data')
if not os.path.exists(savepath):
    os.mkdirs(savepath)
plt.figure()
plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title('Original')
plt.subplot(222)
plt.imshow(noisy, cmap='gray')
plt.axis('off')
plt.title('Noisy \nPSNR={:.2f}, SSMI={:.2f}'.format(results_cnn[0][0], results_cnn[0][1]))
plt.subplot(224)
plt.imshow(denoised_cnn, cmap='gray')
plt.axis('off')
plt.title('CNN {:.2f}sec \nPSNR={:.2f}, SSMI={:.2f}'.format(results_cnn[2], results_cnn[1][0], results_cnn[1][1]))
plt.subplots_adjust(left=0.11, right=0.9, top=0.91, bottom=0.02, hspace=0.25, wspace=0.2)
plt.savefig(os.path.join(savepath, 'CNN_2D.png'), dpi=300)
plt.show()
