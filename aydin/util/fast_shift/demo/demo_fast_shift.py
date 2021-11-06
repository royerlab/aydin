# flake8: noqa
import numpy
from scipy.ndimage import shift

from aydin.util.fast_shift.fast_shift import fast_shift
from aydin.io.datasets import fibsem, examples_single
from aydin.util.log.log import lsection, Log


def demo_fast_shift(image_name: str, image, _shift, repeats=32):
    Log.enable_output = True

    with lsection(f"shift={_shift}:"):
        with lsection(f"Numba-CPU {image_name} (r={repeats}):"):
            for _ in range(repeats):
                numba_shifted_image = fast_shift(image, shift=_shift)

        with lsection(f"Scipy {image_name} (r={repeats}):"):
            for _ in range(repeats):
                scipy_shifted_image = shift(image, shift=_shift)

    # import napari
    #
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(
    #         numba_shifted_image, name='numba_shifted_image', colormap='plasma'
    #     )
    #     viewer.add_image(
    #         scipy_shifted_image, name='scipy_shifted_image', colormap='plasma'
    #     )
    #     viewer.add_image(
    #         numpy.abs(numba_shifted_image - scipy_shifted_image),
    #         name='numba_shifted_image - scipy_shifted_image',
    #         colormap='plasma',
    #     )
    #
    # numpy.testing.assert_array_almost_equal(
    #     numba_shifted_image, scipy_shifted_image, decimal=3
    # )


image_2d = fibsem(full=True).astype(numpy.float32)
image_3d = examples_single.royerlab_hcr.get_array().squeeze()[2]
image_4d = examples_single.hyman_hela.get_array().squeeze()


for _shift in [(3, -1), (17, -51), (-317, 511)]:
    demo_fast_shift("fibsem 2D", image_2d, _shift=_shift, repeats=1)

for _shift in [(3, -1, 2), (17, -51, 7), (-317, 511, -128)]:
    demo_fast_shift("islet 3D", image_3d, _shift=_shift, repeats=1)

for _shift in [(3, -1, 2, -7), (17, -51, -20, 10), (-317, 511, -126, 212)]:
    demo_fast_shift("hela 4D", image_4d, _shift=_shift, repeats=1)
