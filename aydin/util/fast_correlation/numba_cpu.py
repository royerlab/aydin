import numpy
from numba import jit, prange
from numpy.typing import ArrayLike
from scipy.ndimage import correlate as scipy_ndimage_correlate

__fastmath = {'contract', 'afn', 'reassoc'}
__error_model = 'numpy'


def numba_cpu_correlate(image: ArrayLike, kernel: ArrayLike, output=None):
    # Kernel must have odd dimenions:
    if any((s % 2) == 0 for s in kernel.shape):
        raise ValueError(
            "This convolution function only supports kernels with odd lengths."
        )

    # Numba does not support float16 yet:
    dtype = numpy.float32
    image = image.astype(dtype=dtype, copy=False)
    kernel = kernel.astype(dtype=dtype, copy=False)

    # Ensure contiguity:
    if not image.flags['C_CONTIGUOUS']:
        image = numpy.ascontiguousarray(image)
    if not image.flags['C_CONTIGUOUS']:
        kernel = numpy.ascontiguousarray(kernel)

    if output is None:
        output = numpy.zeros_like(image)

    # Switching on a per-dimension basis:
    if image.ndim == 1 and kernel.ndim == 1:
        output = _numba_cpu_correlation_1d(image, kernel, output)
    elif image.ndim == 2 and kernel.ndim == 2:
        output = _numba_cpu_correlation_2d(image, kernel, output)
    elif image.ndim == 3 and kernel.ndim == 3:
        output = _numba_cpu_correlation_3d(image, kernel, output)
    elif image.ndim == 4 and kernel.ndim == 4:
        output = _numba_cpu_correlation_4d(image, kernel, output)
    elif image.ndim == 5 and kernel.ndim == 5:
        output = _numba_cpu_correlation_5d(image, kernel, output)
    elif image.ndim == 6 and kernel.ndim == 6:
        output = _numba_cpu_correlation_6d(image, kernel, output)
    else:
        return scipy_ndimage_correlate(image, kernel, output)

    return output


@jit(nopython=True, parallel=True, error_model=__error_model, fastmath=__fastmath)
def _numba_cpu_correlation_1d(
    image: ArrayLike, kernel: ArrayLike, output: ArrayLike = None
):

    (il0,) = image.shape
    (kl0,) = kernel.shape
    khl0 = kl0 // 2

    def image_get(u):
        u = 0 if u < 0 else u
        u = il0 - 1 if u >= il0 else u
        return image[u]

    for oi0 in prange(il0):
        acc = 0
        for ki0 in range(-khl0, khl0 + 1):
            imgval = image_get(oi0 + ki0)
            kerval = kernel[khl0 + ki0]
            acc += imgval * kerval
        output[oi0] = acc

    return output


@jit(nopython=True, parallel=True, error_model=__error_model, fastmath=__fastmath)
def _numba_cpu_correlation_2d(
    image: ArrayLike, kernel: ArrayLike, output: ArrayLike = None
):

    il0, il1 = image.shape
    kl0, kl1 = kernel.shape
    khl0, khl1 = kl0 // 2, kl1 // 2

    def image_get(u, v):
        u = 0 if u < 0 else u
        u = il0 - 1 if u >= il0 else u
        v = 0 if v < 0 else v
        v = il1 - 1 if v >= il1 else v
        return image[u, v]

    for oi0 in prange(il0):
        for oi1 in range(il1):
            acc = 0
            for ki0 in range(-khl0, khl0 + 1):
                for ki1 in range(-khl1, khl1 + 1):
                    imgval = image_get(oi0 + ki0, oi1 + ki1)
                    kerval = kernel[khl0 + ki0, khl1 + ki1]
                    acc += imgval * kerval
            output[oi0, oi1] = acc

    return output


@jit(nopython=True, parallel=True, error_model=__error_model, fastmath=__fastmath)
def _numba_cpu_correlation_3d(
    image: ArrayLike, kernel: ArrayLike, output: ArrayLike = None
):

    il0, il1, il2 = image.shape
    kl0, kl1, kl2 = kernel.shape
    khl0, khl1, khl2 = kl0 // 2, kl1 // 2, kl2 // 2

    def image_get(u, v, w):
        u = 0 if u < 0 else u
        u = il0 - 1 if u >= il0 else u
        v = 0 if v < 0 else v
        v = il1 - 1 if v >= il1 else v
        w = 0 if w < 0 else w
        w = il2 - 1 if w >= il2 else w
        return image[u, v, w]

    for oi0 in prange(il0):
        for oi1 in range(il1):
            for oi2 in range(il2):
                acc = 0
                for ki0 in range(-khl0, khl0 + 1):
                    for ki1 in range(-khl1, khl1 + 1):
                        for ki2 in range(-khl2, khl2 + 1):
                            imgval = image_get(oi0 + ki0, oi1 + ki1, oi2 + ki2)
                            kerval = kernel[khl0 + ki0, khl1 + ki1, khl2 + ki2]
                            acc += imgval * kerval
                output[oi0, oi1, oi2] = acc

    return output


@jit(nopython=True, parallel=True, error_model=__error_model, fastmath=__fastmath)
def _numba_cpu_correlation_4d(
    image: ArrayLike, kernel: ArrayLike, output: ArrayLike = None, parallelism: int = 8
):

    il0, il1, il2, il3 = image.shape
    kl0, kl1, kl2, kl3 = kernel.shape
    khl0, khl1, khl2, khl3 = kl0 // 2, kl1 // 2, kl2 // 2, kl3 // 2

    def image_get(u, v, w, x):
        u = 0 if u < 0 else u
        u = il0 - 1 if u >= il0 else u
        v = 0 if v < 0 else v
        v = il1 - 1 if v >= il1 else v
        w = 0 if w < 0 else w
        w = il2 - 1 if w >= il2 else w
        x = 0 if x < 0 else x
        x = il3 - 1 if x >= il3 else x
        return image[u, v, w, x]

    for oi0 in prange(il0):
        for oi1 in range(il1):
            for oi2 in range(il2):
                for oi3 in range(il3):
                    acc = 0
                    for ki0 in range(-khl0, khl0 + 1):
                        for ki1 in range(-khl1, khl1 + 1):
                            for ki2 in range(-khl2, khl2 + 1):
                                for ki3 in range(-khl3, khl3 + 1):
                                    imgval = image_get(
                                        oi0 + ki0, oi1 + ki1, oi2 + ki2, oi3 + ki3
                                    )
                                    kerval = kernel[
                                        khl0 + ki0, khl1 + ki1, khl2 + ki2, khl3 + ki3
                                    ]
                                    acc += imgval * kerval
                    output[oi0, oi1, oi2, oi3] = acc

    return output


@jit(nopython=True, parallel=True, error_model=__error_model, fastmath=__fastmath)
def _numba_cpu_correlation_5d(
    image: ArrayLike, kernel: ArrayLike, output: ArrayLike = None, parallelism: int = 8
):

    il0, il1, il2, il3, il4 = image.shape
    kl0, kl1, kl2, kl3, kl4 = kernel.shape
    khl0, khl1, khl2, khl3, khl4 = kl0 // 2, kl1 // 2, kl2 // 2, kl3 // 2, kl4 // 2

    def image_get(u, v, w, x, y):
        u = 0 if u < 0 else u
        u = il0 - 1 if u >= il0 else u
        v = 0 if v < 0 else v
        v = il1 - 1 if v >= il1 else v
        w = 0 if w < 0 else w
        w = il2 - 1 if w >= il2 else w
        x = 0 if x < 0 else x
        x = il3 - 1 if x >= il3 else x
        y = 0 if y < 0 else y
        y = il4 - 1 if y >= il4 else y
        return image[u, v, w, x, y]

    for oi0 in prange(il0):
        for oi1 in range(il1):
            for oi2 in range(il2):
                for oi3 in range(il3):
                    for oi4 in range(il4):
                        acc = 0
                        for ki0 in range(-khl0, khl0 + 1):
                            for ki1 in range(-khl1, khl1 + 1):
                                for ki2 in range(-khl2, khl2 + 1):
                                    for ki3 in range(-khl3, khl3 + 1):
                                        for ki4 in range(-khl4, khl4 + 1):
                                            imgval = image_get(
                                                oi0 + ki0,
                                                oi1 + ki1,
                                                oi2 + ki2,
                                                oi3 + ki3,
                                                oi4 + ki4,
                                            )
                                            kerval = kernel[
                                                khl0 + ki0,
                                                khl1 + ki1,
                                                khl2 + ki2,
                                                khl3 + ki3,
                                                khl4 + ki4,
                                            ]
                                            acc += imgval * kerval
                        output[oi0, oi1, oi2, oi3, oi4] = acc

    return output


@jit(nopython=True, parallel=True, error_model=__error_model, fastmath=__fastmath)
def _numba_cpu_correlation_6d(
    image: ArrayLike, kernel: ArrayLike, output: ArrayLike = None, parallelism: int = 8
):

    il0, il1, il2, il3, il4, il5 = image.shape
    kl0, kl1, kl2, kl3, kl4, kl5 = kernel.shape
    khl0, khl1, khl2, khl3, khl4, khl5 = (
        kl0 // 2,
        kl1 // 2,
        kl2 // 2,
        kl3 // 2,
        kl4 // 2,
        kl5 // 2,
    )

    def image_get(u, v, w, x, y, z):
        u = 0 if u < 0 else u
        u = il0 - 1 if u >= il0 else u
        v = 0 if v < 0 else v
        v = il1 - 1 if v >= il1 else v
        w = 0 if w < 0 else w
        w = il2 - 1 if w >= il2 else w
        x = 0 if x < 0 else x
        x = il3 - 1 if x >= il3 else x
        y = 0 if y < 0 else y
        y = il4 - 1 if y >= il4 else y
        z = 0 if z < 0 else z
        z = il5 - 1 if z >= il5 else z
        return image[u, v, w, x, y, z]

    for oi0 in prange(il0):
        for oi1 in range(il1):
            for oi2 in range(il2):
                for oi3 in range(il3):
                    for oi4 in range(il4):
                        for oi5 in range(il5):
                            acc = 0
                            for ki0 in range(-khl0, khl0 + 1):
                                for ki1 in range(-khl1, khl1 + 1):
                                    for ki2 in range(-khl2, khl2 + 1):
                                        for ki3 in range(-khl3, khl3 + 1):
                                            for ki4 in range(-khl4, khl4 + 1):
                                                for ki5 in range(-khl5, khl5 + 1):
                                                    imgval = image_get(
                                                        oi0 + ki0,
                                                        oi1 + ki1,
                                                        oi2 + ki2,
                                                        oi3 + ki3,
                                                        oi4 + ki4,
                                                        oi5 + ki5,
                                                    )
                                                    kerval = kernel[
                                                        khl0 + ki0,
                                                        khl1 + ki1,
                                                        khl2 + ki2,
                                                        khl3 + ki3,
                                                        khl4 + ki4,
                                                        khl5 + ki5,
                                                    ]
                                                    acc += imgval * kerval
                            output[oi0, oi1, oi2, oi3, oi4, oi5] = acc

    return output
