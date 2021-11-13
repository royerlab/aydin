# flake8: noqa
import time

import numpy
from skimage.exposure import rescale_intensity
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.util import random_noise

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import newyork
from aydin.regression.lgbm import LGBMRegressor
from aydin.util.log.log import lprint, lsection


def normalise(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


class Timer:
    def __enter__(self):
        self.start = time.process_time()
        return self

    def __exit__(self, *args):
        self.end = time.process_time()
        self.interval = self.end - self.start


def demo_opencl_inference():

    image = newyork().astype(numpy.float32)
    image = normalise(image)

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(numpy.float32)

    generator = StandardFeatureGenerator()
    regressor = LGBMRegressor(max_num_estimators=300)

    def callback(iteration, val_loss, model):
        lprint(f"iteration={iteration}, val_loss={val_loss}")

    lprint('feature generation...')
    features = generator.compute(noisy, exclude_center_value=True)

    lprint('reshaping...')
    x = features.reshape(-1, features.shape[-1])
    y = noisy.reshape(-1)

    lprint(f"Number of data points             : {x.shape[0]}")
    lprint(f"Number of features per data points: {x.shape[-1]}")

    lprint('fitting...')
    regressor.fit(x, y, x_valid=x, y_valid=y, regressor_callback=callback)

    lprint('feature generation for inference -- with center value --...')
    features = generator.compute(noisy, exclude_center_value=False)
    x = features.reshape(-1, features.shape[-1])

    for i in range(1, 2):
        with lsection(f'predicting gpu ({i})...'):
            regressor.gpu_prediction = True
            with Timer() as t:
                yp_gpu = regressor.predict(x)
            print(f' ... took {t.interval} seconds')

    ## uncomment to compare speed (typicsally gpu = 20x faster than cpu)
    # print(f'predicting cpu ... ')
    # regressor.gpu_prediction = False
    # with Timer() as t:
    #     yp_cpu = regressor.predict(x)
    # print(f' ... took {t.interval} seconds')
    yp_cpu = yp_gpu * 0.01  # place holder array so that rest of the code runs

    denoised_cpu = yp_cpu.reshape(image.shape)
    denoised_cpu = numpy.clip(denoised_cpu, 0, 1)
    denoised_gpu = yp_gpu.reshape(image.shape)
    denoised_gpu = numpy.clip(denoised_gpu, 0, 1)

    ssim_value_cpu = ssim(denoised_cpu, image)
    psnr_value_cpu = psnr(denoised_cpu, image)
    ssim_value_gpu = ssim(denoised_gpu, image)
    psnr_value_gpu = psnr(denoised_gpu, image)

    lprint("denoised cpu", psnr_value_cpu, ssim_value_cpu)
    lprint("denoised gpu", psnr_value_gpu, ssim_value_gpu)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(normalise(image), name='image')
        viewer.add_image(normalise(noisy), name='noisy')
        viewer.add_image(normalise(denoised_cpu), name='denoised cpu')
        viewer.add_image(normalise(denoised_gpu), name='denoised gpu')


demo_opencl_inference()
