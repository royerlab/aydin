"""Demonstrate 3D Noise2Self denoising with CatBoost on HCR data.

This demo applies self-supervised FGR denoising with CatBoost to a
RoyerLab HCR (hybridization chain reaction) dataset, using variance
stabilisation, range normalisation, and padding transforms, with napari
visualization.
"""

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
    """Denoise a 3D HCR volume using FGR with CatBoost and VST transforms.

    Parameters
    ----------
    image : numpy.ndarray
        Input 3D noisy image array from HCR dataset.
    """

    Log.enable_output = True
    # Log.set_log_max_depth(5)

    generator = StandardFeatureGenerator(include_spatial_features=True)
    regressor = CBRegressor(patience=20, gpu=True)

    it = ImageTranslatorFGR(
        feature_generator=generator, regressor=regressor, balance_training_data=True
    )

    it.add_transform(VarianceStabilisationTransform())
    it.add_transform(RangeTransform())
    it.add_transform(PaddingTransform(pad_width=16))

    it.train(image, image)
    denoised = it.translate(image)

    import napari

    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(denoised, name='denoised')
    napari.run()


if __name__ == "__main__":
    hcr = examples_single.royerlab_hcr.get_array().squeeze()
    hcr = hcr[1]

    demo(hcr)
