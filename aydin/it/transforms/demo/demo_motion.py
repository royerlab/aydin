from pprint import pprint

import numpy
from skimage.util import random_noise

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import normalise, pollen
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.motion import MotionStabilisationTransform
from aydin.it.transforms.range import RangeTransform
from aydin.regression.lgbm import LGBMRegressor
from aydin.util.log.log import Log


def demo_motion():

    Log.enable_output = True

    shifts = tuple((5 * i, int(0.5 * i * i)) for i in range(10))

    print('')
    pprint(shifts)

    image = normalise(pollen())[0:256, 0:256]
    array = numpy.stack(
        [add_noise(numpy.roll(image, shift=shift, axis=(0, 1))) for shift in shifts]
    )

    mc = MotionStabilisationTransform(axes=0)

    preprocessed_array = mc.preprocess(array.copy())
    postprocessed_array = mc.postprocess(preprocessed_array.copy())

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(array, name='array')
        viewer.add_image(preprocessed_array, name='corrected_array')
        viewer.add_image(postprocessed_array, name='uncorrected_array')

    # assert not (array == corrected_array).all()
    assert (array == postprocessed_array).all()

    generator = StandardFeatureGenerator(
        include_corner_features=True,
        include_scale_one=True,
        include_fine_features=True,
        include_spatial_features=True,
    )
    regressor = LGBMRegressor(patience=128, compute_training_loss=True)

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)
    it.add_transform(RangeTransform())
    it.train(array)
    denoised_without_preprocessing = it.translate(array)

    it.add_transform.append(mc)
    it.train(array)
    denoised_with_preprocessing = it.translate(array)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(array, name='array')
        viewer.add_image(preprocessed_array, name='preprocessed_array')
        viewer.add_image(postprocessed_array, name='postprocessed_array')
        viewer.add_image(
            denoised_without_preprocessing, name='denoised_without_preprocessing'
        )
        viewer.add_image(
            denoised_with_preprocessing, name='denoised_with_preprocessing'
        )


def add_noise(image, intensity=4, variance=0.4):
    noisy = image
    if intensity is not None:
        noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode="gaussian", var=variance)
    noisy = noisy.astype(numpy.float32, copy=False)
    return noisy


demo_motion()
