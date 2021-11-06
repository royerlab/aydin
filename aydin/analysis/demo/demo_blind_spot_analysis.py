from aydin.analysis.blind_spot_analysis import auto_detect_blindspots
from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import examples_single
from aydin.it.deconvolution.lr_deconv_scipy import ImageTranslatorLRDeconvScipy
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.range import RangeTransform
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log


def demo_blind_spot_analysis():
    Log.enable_output = True

    image = examples_single.myers_tribolium.get_array()

    blind_spots, noise_auto = auto_detect_blindspots(image)

    # Here are the blind spots that should be used with N2S:
    print(blind_spots)

    lr = ImageTranslatorLRDeconvScipy(psf_kernel=noise_auto, max_num_iterations=100)
    lr.add_transform(RangeTransform())
    lr.train(image)
    deconvolved_image = lr.translate(image)

    generator = StandardFeatureGenerator(
        include_corner_features=True,
        include_scale_one=True,
        include_fine_features=True,
        include_spatial_features=True,
    )

    regressor = CBRegressor(max_num_estimators=1024, patience=32, gpu=True)

    it = ImageTranslatorFGR(
        feature_generator=generator, regressor=regressor, blind_spots='auto'
    )
    it.add_transform(RangeTransform())
    it.train(image)
    denoised_image = it.translate(image)

    it.train(deconvolved_image)
    denoised_deconvolved_image = it.translate(deconvolved_image)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(denoised_image, name='denoised_image')
        viewer.add_image(deconvolved_image, name='deconvolved_image')
        viewer.add_image(denoised_deconvolved_image, name='denoised_deconvolved_image')
        viewer.add_image(noise_auto, name='noise_auto')


demo_blind_spot_analysis()
