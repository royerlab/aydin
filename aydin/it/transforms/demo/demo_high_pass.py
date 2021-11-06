# flake8: noqa
import numpy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import random_noise

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import normalise, camera
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.highpass import HighpassTransform
from aydin.it.transforms.range import RangeTransform
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log


def demo_high_pass_real():

    Log.override_test_exclusion = True
    Log.enable_output = True

    # #Salt&Pepper:
    # image = normalise(camera().astype(numpy.float32))
    # noisy = image
    # noisy = random_noise(noisy, mode="s&p", amount=0.1)

    # #Gaussian:
    # image = normalise(camera().astype(numpy.float32))
    # noisy = image
    # noisy = random_noise(noisy, mode="gaussian", var=0.001)

    # Poisson:
    image = camera() // 5

    noisy = image
    noisy = random_noise(noisy, mode="poisson", clip=False).astype(numpy.float32)
    noisy = normalise(noisy)

    ac = HighpassTransform()

    preprocessed = ac.preprocess(noisy)
    postprocessed = ac.postprocess(preprocessed)

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
    regressor = CBRegressor(patience=16)

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)
    it.add_transform(RangeTransform())
    it.train(noisy)
    denoised_without_preprocessing = it.translate(noisy)

    image = numpy.clip(image, 0, 1)
    denoised_without_preprocessing = numpy.clip(denoised_without_preprocessing, 0, 1)

    psnr_denoised_without = peak_signal_noise_ratio(
        image, denoised_without_preprocessing
    )
    ssim_denoised_without = structural_similarity(image, denoised_without_preprocessing)

    results = []
    results.append(
        f"ssim_denoised_without={ssim_denoised_without}, psnr_denoised_without={psnr_denoised_without}"
    )

    denoised_images_with_preprocessing = []
    low_pass_images = []
    preprocessed_images = []

    for i in range(10):
        scale = 1 + i
        transform = HighpassTransform(scale, median_filtering=True)
        low_pass_images.append(transform._low_pass_filtering(noisy))
        preprocessed_images.append(transform.preprocess(noisy))

        it.clear_transforms()
        it.add_transform(RangeTransform())
        it.add_transform(transform)
        it.train(noisy)
        denoised_image_with_preprocessing = numpy.clip(it.translate(noisy), 0, 1)
        denoised_images_with_preprocessing.append(denoised_image_with_preprocessing)

        psnr_denoised_with = peak_signal_noise_ratio(
            image, denoised_image_with_preprocessing
        )
        ssim_denoised_with = structural_similarity(
            image, denoised_image_with_preprocessing
        )
        results.append(
            f"scale={scale}, ssim_denoised_with={ssim_denoised_with}, psnr_denoised_with={psnr_denoised_with}"
        )

    for result in results:
        print(result)

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
            numpy.stack(denoised_images_with_preprocessing),
            name='denoised_images_with_preprocessing',
        )

        viewer.add_image(numpy.stack(preprocessed_images), name='preprocessed_images')

        viewer.add_image(numpy.stack(low_pass_images), name='low_pass_images')


demo_high_pass_real()
