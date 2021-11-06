# flake8: noqa


from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import examples_single
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log


def demo(image):
    """ """

    Log.enable_output = True
    # Log.set_log_max_depth(5)

    generator = StandardFeatureGenerator()
    regressor = CBRegressor(patience=20, gpu=True)

    it = ImageTranslatorFGR(feature_generator=generator, regressor=regressor)

    it.add_transform(VarianceStabilisationTransform())
    it.add_transform(RangeTransform())
    it.add_transform(PaddingTransform(pad_width=16))

    it.train(image, image)
    denoised = it.translate(image)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(denoised, name='denoised')


islet = examples_single.royerlab_hcr.get_array().squeeze()
# import napari
# with napari.gui_qt():
#     viewer = napari.Viewer()
#     viewer.add_image(islet, name='islet')
# islet = islet[2, :20, 400 : 400 + 256, 700 : 700 + 256]
islet = islet[1]

demo(islet)
