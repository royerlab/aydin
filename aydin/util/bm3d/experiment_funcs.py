"""Utility functions for BM3D denoising experiments.

Provides functions for computing PSNR, generating noise kernels, and
creating noise realizations for BM3D benchmark experiments.
"""

import numpy as np
from bm3d import gaussian_kernel
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
from scipy.signal import fftconvolve


def get_psnr(y_est: np.ndarray, y_ref: np.ndarray) -> float:
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Assumes the noise-free signal maximum is 1.

    Parameters
    ----------
    y_est : numpy.ndarray
        Estimated (denoised) image.
    y_ref : numpy.ndarray
        Noise-free reference image.

    Returns
    -------
    float
        PSNR value in decibels.
    """
    return 10 * np.log10(1 / np.mean(((y_est - y_ref).ravel()) ** 2))


def get_cropped_psnr(y_est: np.ndarray, y_ref: np.ndarray, crop: tuple) -> float:
    """Compute PSNR after cropping border regions from both images.

    Assumes the noise-free signal maximum is 1. Crops are applied
    symmetrically from both sides of each spatial dimension.

    Parameters
    ----------
    y_est : numpy.ndarray
        Estimated (denoised) image.
    y_ref : numpy.ndarray
        Noise-free reference image.
    crop : tuple of int
        Number of pixels to crop from each side along (x, y) dimensions.

    Returns
    -------
    float
        PSNR value in decibels for the cropped region.
    """
    return get_psnr(
        np.atleast_3d(y_est)[crop[0] : -crop[0], crop[1] : -crop[1], :],
        np.atleast_3d(y_ref)[crop[0] : -crop[0], crop[1] : -crop[1], :],
    )


def get_experiment_kernel(
    noise_type: str, noise_var: float, sz: tuple = np.array((101, 101))
):
    """Generate a noise correlation kernel for a specific experiment type.

    Constructs a convolution kernel whose L2 norm equals the square root
    of the specified noise variance. Supports various spatially correlated
    noise patterns from the BM3D literature.

    Parameters
    ----------
    noise_type : str
        Noise type identifier. Accepted values: 'gw', 'g0' (white noise),
        'g1' (horizontal line), 'g2' (circular pattern), 'g3' (diagonal
        line pattern), 'g4' (pink noise). Append 'w' (e.g. 'g1w') to add
        a white noise component.
    noise_var : float
        Desired noise variance.
    sz : tuple of int, optional
        Image size, used only for 'g4' and 'g4w' noise types.
        Defaults to (101, 101).

    Returns
    -------
    numpy.ndarray
        Noise correlation kernel normalized so that its L2 norm equals
        ``sqrt(noise_var)``.

    Raises
    ------
    ValueError
        If ``noise_type`` is not one of the supported types.
    """
    # if noiseType == gw / g0
    kernel = np.array([[1]])
    noise_types = ['gw', 'g0', 'g1', 'g2', 'g3', 'g4', 'g1w', 'g2w', 'g3w', 'g4w']
    if noise_type not in noise_types:
        raise ValueError("Noise type must be one of " + str(noise_types))

    if noise_type != "g4" and noise_type != "g4w":
        # Crop this size of kernel when generating,
        # unless pink noise, in which
        # if noiseType == we want to use the full image size
        sz = np.array([101, 101])
    else:
        sz = np.array(sz)

    # Sizes for meshgrids
    sz2 = -(1 - (sz % 2)) * 1 + np.floor(sz / 2)
    sz1 = np.floor(sz / 2)
    uu, vv = np.meshgrid(
        [i for i in range(-int(sz1[0]), int(sz2[0]) + 1)],
        [i for i in range(-int(sz1[1]), int(sz2[1]) + 1)],
    )

    beta = 0.8

    if noise_type[0:2] == 'g1':
        # Horizontal line
        kernel = np.atleast_2d(16 - abs(np.linspace(1, 31, 31) - 16))

    elif noise_type[0:2] == 'g2':
        # Circular repeating pattern
        scale = 1
        dist = uu**2 + vv**2
        kernel = np.cos(np.sqrt(dist) / scale) * gaussian_kernel((sz[0], sz[1]), 10)

    elif noise_type[0:2] == 'g3':
        # Diagonal line pattern kernel
        scale = 1
        kernel = np.cos((uu + vv) / scale) * gaussian_kernel((sz[0], sz[1]), 10)

    elif noise_type[0:2] == 'g4':
        # Pink noise
        dist = uu**2 + vv**2
        n = sz[0] * sz[1]
        spec = np.sqrt((np.sqrt(n) * 1e-2) / (np.sqrt(dist) + np.sqrt(n) * 1e-2))
        kernel = fftshift(ifft2(ifftshift(spec)))

    else:  # gw and g0 are white
        beta = 0

    # -- Noise with additional white component --

    if len(noise_type) > 2 and noise_type[2] == 'w':
        kernel = kernel / np.sqrt(np.sum(kernel**2))
        kalpha = np.sqrt((1 - beta) + beta * abs(fft2(kernel, (sz[0], sz[1]))) ** 2)
        kernel = fftshift(ifft2(kalpha))

    kernel = np.real(kernel)
    # Correct variance
    kernel = kernel / np.sqrt(np.sum(kernel**2)) * np.sqrt(noise_var)

    return kernel


def get_experiment_noise(
    noise_type: str, noise_var: float, realization: int, sz: tuple
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Generate spatially correlated noise for a BM3D experiment.

    Creates non-circular noise by convolving white Gaussian noise with
    the experiment kernel and cropping edges to avoid boundary artifacts.

    Parameters
    ----------
    noise_type : str
        Noise type identifier. See ``get_experiment_kernel`` for accepted values.
    noise_var : float
        Desired noise variance.
    realization : int
        Random seed for reproducible noise generation.
    sz : tuple of int
        Image size determining the shape of the output noise array.

    Returns
    -------
    noise : numpy.ndarray
        Generated noise array with shape ``sz``.
    psd : numpy.ndarray
        Power spectral density of the noise.
    kernel : numpy.ndarray
        Correlation kernel used to generate the noise.
    """
    np.random.seed(realization)

    # Get pre-specified kernel
    kernel = get_experiment_kernel(noise_type, noise_var, sz)

    # Create noisy image
    half_kernel = np.ceil(np.array(kernel.shape) / 2)

    if len(sz) == 3 and half_kernel.size == 2:
        half_kernel = [half_kernel[0], half_kernel[1], 0]
        kernel = np.atleast_3d(kernel)

    half_kernel = np.array(half_kernel, dtype=int)

    # Crop edges
    noise = fftconvolve(
        np.random.normal(size=(sz + 2 * half_kernel)), kernel, mode='same'
    )
    noise = np.atleast_3d(noise)[
        half_kernel[0] : -half_kernel[0], half_kernel[1] : -half_kernel[1], :
    ]

    psd = abs(fft2(kernel, (sz[0], sz[1]), axes=(0, 1))) ** 2 * sz[0] * sz[1]

    return noise, psd, kernel
