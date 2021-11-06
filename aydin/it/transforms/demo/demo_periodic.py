import numpy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import normalise, camera, add_noise
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.periodic import PeriodicNoiseSuppressionTransform
from aydin.it.transforms.range import RangeTransform
from aydin.regression.lgbm import LGBMRegressor
from aydin.util.log.log import Log


def demo_high_pass_real():

    Log.override_test_exclusion = True
    Log.enable_output = True

    image = normalise(camera().astype(numpy.float32))

    noisy = image.copy()

    freq = 96
    periodic_pattern = 0.1 * (
        1 + numpy.cos(numpy.linspace(0, freq * 2 * numpy.pi, num=image.shape[0]))
    )
    periodic_pattern = periodic_pattern[:, numpy.newaxis]
    noisy += periodic_pattern

    noisy = add_noise(noisy)

    pns = PeriodicNoiseSuppressionTransform()

    preprocessed = pns.preprocess(noisy)
    postprocessed = pns.postprocess(preprocessed)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(preprocessed, name='preprocessed')
        viewer.add_image(postprocessed, name='postprocessed')

    generator = StandardFeatureGenerator(
        include_corner_features=True,
        include_scale_one=True,
        include_fine_features=True,
        include_spatial_features=True,
    )
    regressor = LGBMRegressor(
        patience=128, compute_training_loss=True, max_num_estimators=2048
    )
    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)
    it.add_transform(RangeTransform())

    it.train(noisy)
    denoised_without_preprocessing = it.translate(noisy)

    it.add_transform(pns)
    it.train(noisy)
    denoised_with_preprocessing = it.translate(noisy)

    image = numpy.clip(image, 0, 1)
    denoised_without_preprocessing = numpy.clip(denoised_without_preprocessing, 0, 1)
    denoised_with_preprocessing = numpy.clip(denoised_with_preprocessing, 0, 1)
    psnr_denoised_without = psnr(image, denoised_without_preprocessing)
    ssim_denoised_without = ssim(image, denoised_without_preprocessing)
    psnr_denoised_with = psnr(image, denoised_with_preprocessing)
    ssim_denoised_with = ssim(image, denoised_with_preprocessing)
    print("denoised_without:", psnr_denoised_without, ssim_denoised_without)
    print("denoised_with   :", psnr_denoised_with, ssim_denoised_with)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(preprocessed, name='preprocessed')
        viewer.add_image(postprocessed, name='postprocessed')
        viewer.add_image(
            denoised_without_preprocessing, name='denoised_without_preprocessing'
        )
        viewer.add_image(
            denoised_with_preprocessing, name='denoised_with_preprocessing'
        )


demo_high_pass_real()
