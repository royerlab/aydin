# flake8: noqa
import time

from skimage.exposure import rescale_intensity

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log


def demo():
    Log.enable_output = True

    image_path = examples_single.hyman_hela.get_path()
    image, metadata = io.imread(image_path)
    print(image.shape)

    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    generator = StandardFeatureGenerator(max_level=7, include_spatial_features=True)
    regressor = CBRegressor()

    it = ImageTranslatorFGR(
        generator,
        regressor,
        balance_training_data=True,
        max_voxels_for_training=10_000_000,
    )

    batch_dims = (False, False, False, False)

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


if __name__ == "__main__":
    demo()
