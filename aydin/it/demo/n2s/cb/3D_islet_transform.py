# flake8: noqa

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import examples_single
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.attenuation import AttenuationTransform
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log


def demo(image):
    """
    In some cases it might be usefull to append a compression transform (sqrt) after normalisation,
    something akin to a VST transform but without the exact variance stabilisation, and more as a way
    to deskew the histogram. There are only a few situations where this truly helps, and there are not many.
    So by default this is off.
    """

    Log.enable_output = True
    # Log.set_log_max_depth(5)

    generator = StandardFeatureGenerator(
        # include_scale_one=True,
        # include_fine_features=True,
        # include_corner_features=True,
        # include_line_features=True,
        # decimate_large_scale_features=False,
        # extend_large_scale_features=True,
        include_corner_features=True,
        include_scale_one=True,
        include_fine_features=True,
        # include_spatial_features=True,
    )
    regressor = CBRegressor(patience=20, gpu=True)

    it = ImageTranslatorFGR(
        feature_generator=generator, regressor=regressor, normaliser_transform='sqrt'
    )

    it.train(image, image)
    denoised = it.translate(image)

    ac = AttenuationTransform(axes=0)
    corrected = ac.preprocess(image)

    it.train(corrected, corrected)
    denoised_corrected = it.translate(corrected)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(corrected, name='noisy')
        viewer.add_image(denoised, name='denoised')
        viewer.add_image(denoised_corrected, name='denoised_corrected')


islet = examples_single.royerlab_hcr.get_array().squeeze()
islet = islet[2, :20, 400 : 400 + 256, 700 : 700 + 256]
demo(islet)
