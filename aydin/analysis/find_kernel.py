import numpy
from numpy.fft import fftn, ifftn, fftshift


def compute_relative_blur_kernel(image_clean, image_blurry, size: int = 3):
    """Compute relative blur kernel.

    Parameters
    ----------
    image_clean : numpy.typing.ArrayLike
    image_blurry : numpy.typing.ArrayLike
    size : int

    Returns
    -------
    kernel : numpy.typing.ArrayLike

    """

    image_blurry_fft = fftn(image_blurry)
    image_clean_fft = fftn(image_clean)
    kernel_fft = image_clean_fft / image_blurry_fft
    kernel = ifftn(kernel_fft)

    radius = (size - 1) // 2
    slice_spec = tuple(
        slice(s // 2 - radius, s // 2 + radius + 1) for s in kernel.shape
    )
    kernel = fftshift(kernel)
    kernel = kernel[slice_spec]
    kernel = numpy.absolute(kernel)
    kernel = kernel / kernel.sum()

    return kernel
