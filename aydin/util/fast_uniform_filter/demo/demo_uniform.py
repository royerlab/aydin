# flake8: noqa
import numpy
from scipy.ndimage import uniform_filter

from aydin.util.fast_uniform_filter.numba_cpu_uf import numba_cpu_uniform_filter
from aydin.util.fast_uniform_filter.numba_gpu_uf import numba_gpu_uniform_filter
from aydin.util.fast_uniform_filter.parallel_uf import parallel_uniform_filter
from aydin.io.datasets import fibsem, examples_single
from aydin.util.log.log import lsection, Log


def demo_par_uniform(image_name: str, image, size=128, repeats=32):
    Log.enable_output = True

    with lsection(f"size={size}:"):
        with lsection(f"Numba-CPU {image_name} (r={repeats}):"):
            for _ in range(repeats):
                numba_filtered_image = numba_cpu_uniform_filter(
                    image, size=size, mode="nearest"
                )

        with lsection(f"Parallel {image_name} (r={repeats}):"):
            for _ in range(repeats):
                parallel_filtered_image = parallel_uniform_filter(image, size=size)

        with lsection(f"Scipy {image_name} (r={repeats}):"):
            for _ in range(repeats):
                scipy_filtered_image = uniform_filter(image, size=size, mode="nearest")

        with lsection(f"Numba-CUDA {image_name} (r={repeats}):"):
            for _ in range(repeats):
                numba_cuda_filtered_image = numba_gpu_uniform_filter(image, size=size)

    # import napari
    #
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(
    #         numba_filtered_image, name='numba_filtered_image', colormap='plasma'
    #     )
    #     viewer.add_image(
    #         parallel_filtered_image, name='parallel_filtered_image', colormap='plasma'
    #     )
    #     viewer.add_image(
    #         scipy_filtered_image, name='scipy_filtered_image', colormap='plasma'
    #     )
    #
    #     viewer.add_image(
    #         numpy.abs(scipy_filtered_image - parallel_filtered_image),
    #         name='scipy_filtered_image - parallel_filtered_image',
    #         colormap='plasma',
    #     )
    #     viewer.add_image(
    #         numpy.abs(scipy_filtered_image - numba_filtered_image),
    #         name='scipy_filtered_image - numba_filtered_image',
    #         colormap='plasma',
    #     )

    numpy.testing.assert_array_almost_equal(
        parallel_filtered_image, scipy_filtered_image, decimal=3
    )

    numpy.testing.assert_array_almost_equal(
        numba_filtered_image, scipy_filtered_image, decimal=3
    )

    numpy.testing.assert_array_almost_equal(
        numba_cuda_filtered_image, scipy_filtered_image, decimal=3
    )


image_2d = fibsem(full=True).astype(numpy.float32)
image_3d = examples_single.royerlab_hcr.get_array().squeeze()[2]
image_4d = examples_single.hyman_hela.get_array().squeeze()

sizes = [3, 9, 64, 127, 317, 511]

for size in sizes:
    demo_par_uniform("fibsem 2D", image_2d, size=size)

for size in sizes:
    demo_par_uniform("islet 3D", image_3d, size=size, repeats=1)

for size in sizes:
    demo_par_uniform("hela 4D", image_4d, size=size, repeats=1)
