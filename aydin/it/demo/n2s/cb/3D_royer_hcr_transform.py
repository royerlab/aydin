"""Demonstrate 3D Noise2Self denoising with CatBoost and attenuation correction on HCR data.

This demo applies self-supervised FGR denoising with CatBoost to a
RoyerLab HCR dataset, comparing results with and without attenuation
correction along the z-axis (useful for light-sheet microscopy data),
with napari visualization.
"""

# flake8: noqa

from aydin.features.standard_features import StandardFeatureGenerator
from aydin.io.datasets import examples_single
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.attenuation import AttenuationTransform
from aydin.regression.cb import CBRegressor
from aydin.util.log.log import Log


def demo(image):
    """Denoise a 3D HCR volume using FGR with CatBoost and attenuation correction.

    Demonstrates the effect of applying an attenuation correction transform
    along the z-axis before denoising, compared to denoising the raw data.
    A sqrt normaliser transform is used to deskew the histogram, which can
    help in some situations (akin to a VST without exact variance
    stabilisation).

    Parameters
    ----------
    image : numpy.ndarray
        Input 3D noisy image array from HCR dataset.
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

    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(corrected, name='noisy')
    viewer.add_image(denoised, name='denoised')
    viewer.add_image(denoised_corrected, name='denoised_corrected')
    napari.run()


if __name__ == "__main__":
    hcr = examples_single.royerlab_hcr.get_array().squeeze()
    hcr = hcr[2, :20, 400 : 400 + 256, 700 : 700 + 256]
    demo(hcr)
