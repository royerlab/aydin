import time

import numpy
import pytest
from skimage.exposure import rescale_intensity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.util import random_noise

from aydin.io.datasets import newyork
from aydin.regression.lgbm import LGBMRegressor


def normalise(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


@pytest.mark.opencl
def test_opencl_inference():
    from aydin.features.fast.fast_features import FastFeatureGenerator

    image = newyork().astype(numpy.float32)
    image = normalise(image)

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(numpy.float32)

    # feature generator requires images in 'standard' form: BCTZYX
    noisy = image[numpy.newaxis, numpy.newaxis, ...]

    generator = FastFeatureGenerator()
    regressor = LGBMRegressor(max_num_estimators=130)

    def callback(iteration, val_loss, model):
        print(f"iteration={iteration}, val_loss={val_loss}")

    print('feature generation...')
    features = generator.compute(noisy, exclude_center_value=True)

    print('reshaping...')
    x = features.reshape(-1, features.shape[-1])
    y = noisy.reshape(-1)

    print(f"Number of data points             : {x.shape[0]}")
    print(f"Number of features per data points: {x.shape[-1]}")

    print('fitting...')
    regressor.fit(x, y, x_valid=x, y_valid=y, regressor_callback=callback)

    print('feature generation for inference -- with center value --...')
    features = generator.compute(noisy, exclude_center_value=False)
    x = features.reshape(-1, features.shape[-1])

    for i in range(1, 20):
        print(f'predicting gpu ({i})...')
        regressor.gpu_prediction = True
        with Timer() as t:
            yp_gpu = regressor.predict(x)
        print(f' ... took {t.interval} seconds')

    # uncomment to compare speed (typically gpu = 20x faster than cpu)
    print('predicting cpu ... ')
    regressor.gpu_prediction = False
    with Timer() as t:
        yp_cpu = regressor.predict(x)
    print(f' ... took {t.interval} seconds')
    # yp_cpu = yp_gpu*0.01 # place holder array so that rest of the code runs

    denoised_cpu = yp_cpu.reshape(image.shape)
    denoised_cpu = numpy.clip(denoised_cpu, 0, 1)
    denoised_gpu = yp_gpu.reshape(image.shape)
    denoised_gpu = numpy.clip(denoised_gpu, 0, 1)

    ssim_value_cpu = ssim(denoised_cpu, image)
    psnr_value_cpu = psnr(denoised_cpu, image)
    ssim_value_gpu = ssim(denoised_gpu, image)
    psnr_value_gpu = psnr(denoised_gpu, image)

    print("denoised cpu", psnr_value_cpu, ssim_value_cpu)
    print("denoised gpu", psnr_value_gpu, ssim_value_gpu)

    assert ssim_value_gpu > 0.50
