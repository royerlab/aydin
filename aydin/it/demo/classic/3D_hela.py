# flake8: noqa
import time

from skimage.exposure import rescale_intensity

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.classic import ImageDenoiserClassic
from aydin.it.classic_denoisers.butterworth import (
    calibrate_denoise_butterworth,
    denoise_butterworth,
)
from aydin.it.classic_denoisers.spectral import (
    calibrate_denoise_spectral,
    denoise_spectral,
)
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log


def demo():
    Log.enable_output = True

    image_path = examples_single.hyman_hela.get_path()
    image, metadata = io.imread(image_path)
    print(image.shape)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))
    # image = image[0:3]

    it = ImageDenoiserClassic(method="spectral")

    it.transforms_list.append(RangeTransform())
    it.transforms_list.append(PaddingTransform())

    batch_dims = (False, True, False, False)

    start = time.time()
    it.train(image, image, batch_axes=batch_dims)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    denoised = it.translate(image, batch_axes=batch_dims)
    stop = time.time()
    print(f"inference train: elapsed time:  {stop - start} ")

    print(image.shape)
    print(denoised.shape)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(denoised, name='denoised')


demo()
