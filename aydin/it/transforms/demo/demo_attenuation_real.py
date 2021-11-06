from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import examples_single
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.attenuation import AttenuationTransform
from aydin.it.transforms.range import RangeTransform
from aydin.regression.lgbm import LGBMRegressor
from aydin.util.log.log import Log


def demo_attenuation_correction_real():

    Log.override_test_exclusion = True
    Log.enable_output = True

    image = examples_single.royerlab_hcr.get_array().squeeze()
    image = image[2, :, :, :]

    ac = AttenuationTransform(axes=0)

    preprocessed = ac.preprocess(image)
    postprocessed = ac.postprocess(preprocessed)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(preprocessed, name='preprocessed')
        viewer.add_image(postprocessed, name='postprocessed')

    generator = StandardFeatureGenerator(
        include_corner_features=True,
        include_scale_one=True,
        include_fine_features=True,
        include_spatial_features=True,
    )
    regressor = LGBMRegressor(patience=128, compute_training_loss=True)

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)
    it.add_transform(RangeTransform())
    it.train(image)
    denoised_without_preprocessing = it.translate(image)

    it.add_transform.append(ac)
    it.train(image)
    denoised_with_preprocessing = it.translate(image)

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


demo_attenuation_correction_real()
