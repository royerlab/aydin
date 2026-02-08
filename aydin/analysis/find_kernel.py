"""Blur kernel estimation between image pairs.

This module provides a function to estimate the relative blur kernel between
a clean and a blurry version of the same image using Fourier deconvolution.
"""

import numpy
from numpy.fft import fftn, fftshift, ifftn


def compute_relative_blur_kernel(image_clean, image_blurry, size: int = 3):
    """Compute the relative blur kernel between a clean and blurry image.

    Estimates the point spread function (PSF) that transforms the clean
    image into the blurry image using Fourier-domain deconvolution,
    then crops the result to the specified kernel size.

    Parameters
    ----------
    image_clean : numpy.typing.ArrayLike
        The sharp reference image.
    image_blurry : numpy.typing.ArrayLike
        The blurred version of the same image, same shape as ``image_clean``.
    size : int
        Side length of the output kernel (must be odd for symmetric kernels).

    Returns
    -------
    kernel : numpy.ndarray
        Estimated blur kernel of shape ``(size,) * ndim``, normalized to sum to 1.
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
