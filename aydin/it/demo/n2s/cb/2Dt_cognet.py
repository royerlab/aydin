# flake8: noqa
import time
from datetime import date
from os.path import join

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.io.folders import get_temp_folder
from aydin.io.io import mapped_tiff
from aydin.it.fgr import ImageTranslatorFGR
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log


def demo():
    Log.enable_output = True

    today = date.today()
    filename = f"result_{today.strftime('%m_%d_%Y_%H_%M_%S')}.tiff"

    image_path = examples_single.cognet_nanotube_400fps.get_path()
    array, metadata = io.imread(image_path)

    # Remove some metadata in the first pixels:
    array[:, 0, 0:4] = array[:, 1, 0:4]

    # crop:
    train = array[0:1024]
    infer = array[0:1024]

    # print(f"Number of distinct features in image: {len(numpy.unique(infer))}")

    print(f"train: {train.shape}, inference:{infer.shape} ")

    batch_dims = (False, False, False)

    # generator = FastFeatureGenerator(include_spatial_features=False)
    generator = StandardFeatureGenerator(
        include_corner_features=True,
        include_scale_one=True,
        include_fine_features=True,
        include_spatial_features=True,
    )

    regressor = CBRegressor(max_num_estimators=2048, patience=20, gpu=True)

    it = ImageTranslatorFGR(generator, regressor, max_voxels_for_training=4e7)

    start = time.time()
    it.train(train, train, batch_axes=batch_dims)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    denoised = it.translate(infer, batch_axes=batch_dims, tile_size=512)
    stop = time.time()
    print(f"Inference: elapsed time:  {stop - start} ")

    output_file = join(get_temp_folder(), filename)
    print(f"Output file: {output_file}")

    # We write the stack to a temp file:
    with mapped_tiff(output_file, infer.shape, infer.dtype) as denoised_tiff:
        print(f"Writing to file: {output_file} ")
        denoised_tiff[...] = denoised[...]

        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(train, name='train')
            viewer.add_image(infer, name='infer')
            print(f"Inference: elapsed time:  {stop - start} ")
            viewer.add_image(denoised, name='denoised')


demo()
