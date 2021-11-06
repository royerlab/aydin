import time
from os.path import join

import numpy
from skimage.exposure import rescale_intensity

from aydin.io import imread
from aydin.io.datasets import examples_single
from aydin.io.folders import get_temp_folder
from aydin.it.normalisers.base import NormaliserBase
from aydin.it.normalisers.identity import IdentityNormaliser
from aydin.it.normalisers.minmax import MinMaxNormaliser
from aydin.it.normalisers.percentile import PercentileNormaliser


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_percentile_saveload():
    saveload(PercentileNormaliser(0))


def test_minmax_saveload():
    saveload(MinMaxNormaliser())


def test_identity_saveload():
    saveload(IdentityNormaliser())


def saveload(normaliser):
    input_path = examples_single.hyman_hela.get_path()
    array, metadata = imread(input_path)
    assert array.dtype == numpy.uint16

    normaliser.calibrate(array)
    print(f"Before normalisation: min,max = {(array.min(), array.max())}")

    temp_file = join(
        get_temp_folder(), "test_normaliser_saveload.json" + str(time.time())
    )
    normaliser.save(temp_file)
    del normaliser

    loaded_normaliser = NormaliserBase.load(temp_file)

    new_array = array.copy()
    normalised_array = loaded_normaliser.normalise(new_array)
    print(
        f"After normalisation: min,max = {(normalised_array.min(), normalised_array.max())}"
    )
    assert normalised_array.dtype == numpy.float32

    denormalised_array = loaded_normaliser.denormalise(normalised_array)
    print(
        f"After denormalisation: min,max = {(denormalised_array.min(), denormalised_array.max())}"
    )
    assert denormalised_array.dtype == numpy.uint16

    assert (
        abs(array.min() - denormalised_array.min()) < 5
        and abs(array.max() - denormalised_array.max()) < 20
    )
