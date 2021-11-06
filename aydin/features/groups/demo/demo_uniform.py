# flake8: noqa
import numpy
from numba import threading_layer
from scipy.ndimage import uniform_filter

from aydin.features.groups.uniform import UniformFeatures
from aydin.io.datasets import examples_single
from aydin.util.log.log import lsection, Log


def demo_uniform(image_name: str, image, repeats=32):
    Log.enable_output = True

    uniform = UniformFeatures()

    size = 3

    # Warmup:
    with lsection(f"Numba {image_name} (r={repeats}) WARMUP!:"):
        numba_filtered_image = uniform._compute_uniform_filter(image, size=size)

    with lsection(f"Numba {image_name} (r={repeats}):"):
        for _ in range(repeats):
            numba_filtered_image = uniform._compute_uniform_filter(image, size=size)

    with lsection(f"Scipy {image_name} (r={repeats}):"):
        for _ in range(repeats):
            scipy_filtered_image = uniform_filter(image, size=size, mode="nearest")

    numpy.testing.assert_array_almost_equal(
        numba_filtered_image, scipy_filtered_image, decimal=3
    )

    print("Threading layer chosen: %s" % threading_layer())


# image = fibsem(full=True).astype(numpy.float32)
# demo_numba_uniform("fibsem 2D", image)

image = examples_single.royerlab_hcr.get_array().squeeze()[0, ..., :320]
demo_uniform("islet 3D", image, repeats=10)
#
# image = examples_single.hyman_hela.get_array().squeeze()
# demo_numba_uniform("hela 4D", image, repeats=1)
