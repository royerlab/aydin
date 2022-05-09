# flake8: noqa
import numpy
from numpy import ones
from scipy.ndimage import correlate

from aydin.io.datasets import examples_single, characters
from aydin.util.fast_correlation.numba_cpu import numba_cpu_correlate
from aydin.util.fast_correlation.parallel import parallel_correlate
from aydin.util.log.log import lsection, Log


def demo_correlation_benchmark(image_name: str, image, size=128, repeats=32):
    Log.enable_output = True

    kernel_shape = (size,) * image.ndim
    kernel = ones(kernel_shape)
    kernel /= kernel.sum()

    with lsection(f"size={size}:"):

        numba_filtered_image = numba_cpu_correlate(image, kernel=kernel)
        with lsection(f"Numba-CPU {image_name} (r={repeats}):"):
            for _ in range(repeats):
                numba_filtered_image = numba_cpu_correlate(image, kernel=kernel)

        with lsection(f"Parallel {image_name} (r={repeats}):"):
            for _ in range(repeats):
                parallel_filtered_image = parallel_correlate(image, kernel=kernel)

        with lsection(f"Scipy {image_name} (r={repeats}):"):
            for _ in range(repeats):
                scipy_filtered_image = correlate(image, weights=kernel)


if __name__ == "__main__":
    image_2d = characters().astype(numpy.float32)
    image_3d = examples_single.royerlab_hcr.get_array().squeeze()[2]
    image_4d = examples_single.hyman_hela.get_array().squeeze()

    sizes = [3, 5, 9, 11, 13, 21, 51]

    for size in sizes:
        demo_correlation_benchmark("characters 2D", image_2d, size=size)

    for size in sizes:
        demo_correlation_benchmark("islet 3D", image_3d, size=size, repeats=1)

    for size in sizes:
        demo_correlation_benchmark("hela 4D", image_4d, size=size, repeats=1)
