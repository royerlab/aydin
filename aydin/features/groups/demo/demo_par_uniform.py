# flake8: noqa
import numpy
from scipy.ndimage import uniform_filter

from aydin.util.fast_uniform_filter import parallel_uniform_filter
from aydin.io.datasets import fibsem, examples_single
from aydin.util.log.log import lsection, Log


def demo_par_uniform(image_name: str, image, repeats=32):
    Log.enable_output = True

    size = 4

    with lsection(f"Par {image_name} (r={repeats}):"):
        for _ in range(repeats):
            par_filtered_image = parallel_uniform_filter(image, size=size)

    with lsection(f"Scipy {image_name} (r={repeats}):"):
        for _ in range(repeats):
            scipy_filtered_image = uniform_filter(image, size=size, mode="nearest")

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(
            par_filtered_image, name='par_filtered_image', colormap='plasma'
        )
        viewer.add_image(
            scipy_filtered_image, name='scipy_filtered_image', colormap='plasma'
        )
        viewer.add_image(
            numpy.abs(scipy_filtered_image - par_filtered_image),
            name='scipy_filtered_image',
            colormap='plasma',
        )

    numpy.testing.assert_array_almost_equal(
        par_filtered_image, scipy_filtered_image, decimal=3
    )


image = fibsem(full=True).astype(numpy.float32)
demo_par_uniform("fibsem 2D", image)

image = examples_single.royerlab_hcr.get_array().squeeze()  # [0, ..., :320]
demo_par_uniform("islet 3D", image, repeats=10)

image = examples_single.hyman_hela.get_array().squeeze()
demo_par_uniform("hela 4D", image, repeats=1)
