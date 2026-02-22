"""Realistic camera noise simulation for scientific imaging.

This module provides functions to simulate realistic camera noise including
shot noise, dark current, gain variations, hot/cold pixels, and cosmic rays.
Useful for generating synthetic noisy images for testing denoising algorithms.
"""

from typing import Optional

import numpy
from numpy.random import RandomState

# from numpy.typing import ArrayLike


def simulate_camera_image(
    photons_per_second,
    exposure_time_s: float = 0.20,
    quantum_efficiency: float = 0.82,
    gain: float = 2.22,
    gain_sigma: float = 0.02,
    gain_column_sigma: float = 0.01,
    offset_mean: float = 1.63,
    offset_sigma: float = 1,
    dark_current: float = 0.04,
    dark_current_sigma: float = 0.01,
    dark_current_column_sigma: float = 0.01,
    min_exposure_dark_current: float = 0.001,
    num_hot_pixels: int = 8,
    num_cold_pixels: int = 8,
    probability_cosmic_ray: float = 1e-6,
    bitdepth: int = 12,
    baseline: int = 100,
    shot_rnd: Optional[RandomState] = None,
    camera_rnd: Optional[RandomState] = None,
    dtype=numpy.int32,
):
    """Simulate a realistic scientific camera image with multiple noise sources.

    Models the full imaging pipeline including photon shot noise, dark current,
    gain variations, hot/cold pixels, cosmic rays, readout offset, and pixel
    saturation. Adapted from Kyle Douglas's blog [1]_ with additional ideas
    from [2]_.

    Parameters
    ----------
    photons_per_second : numpy.typing.ArrayLike
        Image representing the number of photons received on the camera
        per pixel per second.
    exposure_time_s : float
        Exposure time in seconds.
    quantum_efficiency : float
        Quantum efficiency, i.e. conversion factor between photons and
        electrons. Typically in the range [0, 1].
    gain : float
        Conversion factor between electrons and Analog-Digital Units (ADU).
    gain_sigma : float
        Standard deviation of per-pixel gain variation. Not all gains are
        identical across the camera pixels; this parameter controls the
        spread of the gain.
    gain_column_sigma : float
        Standard deviation of per-column gain variation. Each column of
        the detector often has its own electronics that induce another
        source of column-dependent noise.
    offset_mean : float
        Mean value of the pixel amplification offset noise.
    offset_sigma : float
        Standard deviation of the pixel amplification offset noise.
    dark_current : float
        Dark current in electrons per pixel per second, as typically
        reported by manufacturers.
    dark_current_sigma : float
        Standard deviation of per-pixel dark current variation.
    dark_current_column_sigma : float
        Standard deviation of per-column dark current variation.
    min_exposure_dark_current : float
        Minimal effective exposure time for dark current accumulation.
        The effects of dark current do not completely vanish for very
        short exposures.
    num_hot_pixels : int
        Number of hot pixels to simulate.
    num_cold_pixels : int
        Number of cold pixels to simulate.
    probability_cosmic_ray : float
        Probability per pixel per second that a cosmic ray will hit a
        camera pixel.
    bitdepth : int
        Bit depth of each pixel of the camera (e.g., 12 for 12-bit).
    baseline : int
        Baseline ADU value added to the image.
    shot_rnd : numpy.random.RandomState, optional
        Random state for shot-noise generation (time-dependent noise).
        If None, a new unseeded RandomState is created.
    camera_rnd : numpy.random.RandomState, optional
        Random state for camera-specific noise (time-independent, camera
        instance-dependent). If None, a seeded RandomState (seed=42) is
        created for reproducibility.
    dtype : numpy.dtype
        Integer dtype for the output image.

    Returns
    -------
    adu : numpy.ndarray
        Simulated camera image in analog-to-digital units (ADU) with the
        specified dtype and bit depth.

    References
    ----------
    .. [1] Kyle Douglas, "Modeling Noise for Image Simulations",
       http://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/
    .. [2] Matt Craig, "Construction of an artificial but realistic image",
       https://mwcraig.github.io/ccd-as-book/01-03-Construction-of-an-artificial-but-realistic-image.html
    """

    if shot_rnd is None:
        shot_rnd = numpy.random.RandomState(seed=None)

    if camera_rnd is None:
        camera_rnd = numpy.random.RandomState(seed=42)

    # Gain image:
    gain_image = gain * numpy.ones_like(photons_per_second, dtype=numpy.float32)

    # Unfortunately the gain is a bit different for each pixel:
    if gain_sigma > 0:
        gain_image += _normal(
            camera_rnd, scale=gain_sigma, size=photons_per_second.shape
        )

    # And often each column of the detector has its own electronics
    # that induce another source of column-dependent noise:
    if gain_column_sigma > 0:
        gain_image += _normal(
            camera_rnd, scale=gain_column_sigma, size=photons_per_second.shape[1:]
        )[numpy.newaxis, :]

    # Clip gain:
    gain_image = numpy.clip(gain_image, a_min=0, a_max=None, out=gain_image)

    # Readout offset noise:
    offset_image = offset_mean + _normal(
        shot_rnd, scale=offset_sigma, size=photons_per_second.shape
    )

    # Dark current image:
    dark_current_image = dark_current * numpy.ones_like(
        photons_per_second, dtype=numpy.float32
    )

    # Unfortunately the dark current is a bit different for each pixel:
    if dark_current_sigma > 0:
        dark_current_image += _normal(
            camera_rnd, scale=dark_current_sigma, size=photons_per_second.shape
        )

    # And often each column of the detector has its own electronics
    # that induce another source of column-dependent noise:
    if dark_current_column_sigma > 0:
        dark_current_image += _normal(
            camera_rnd,
            scale=dark_current_column_sigma,
            size=photons_per_second.shape[1:],
        )[numpy.newaxis, :]

    # Add shot noise
    photons = _poisson(
        shot_rnd, photons_per_second * exposure_time_s, size=photons_per_second.shape
    )

    # Converts from photons to electrons:
    electrons = quantum_efficiency * photons

    # Epsilon value for clipping lowgain:
    epsilon = 1e-6

    # dark current electrons:
    dark_electrons = _poisson(
        shot_rnd,
        numpy.clip(dark_current_image, a_min=epsilon, a_max=None)
        * max(min_exposure_dark_current, exposure_time_s),
        size=photons.shape,
    )

    # Cosmic rays (lol):
    if probability_cosmic_ray > 0:
        num_of_rays = exposure_time_s * probability_cosmic_ray * electrons.size
        effective_num_of_rays = shot_rnd.poisson(num_of_rays)

        shape = electrons.shape
        ray_indices = tuple(
            camera_rnd.randint(0, s, size=effective_num_of_rays) for s in shape
        )
        dark_electrons[ray_indices] += int(gain * 16)

    # Some pixels are hot:
    if num_hot_pixels > 0:
        shape = dark_electrons.shape
        hot_indices = tuple(
            camera_rnd.randint(0, s, size=num_hot_pixels) for s in shape
        )
        dark_electrons[hot_indices] *= min(16, 2 ** (bitdepth - 2))

    # Some pixels are cold:
    if num_cold_pixels > 0:
        shape = electrons.shape
        cold_indices = tuple(
            camera_rnd.randint(0, s, size=num_cold_pixels) for s in shape
        )
        electrons[cold_indices] /= min(16, 2 ** (bitdepth - 2))

    # Add dark current
    all_electrons = dark_electrons + electrons

    # max ADU:
    max_adu = int(2**bitdepth - 1)

    # Convert to discrete numbers (ADU):
    adu = (all_electrons * gain_image + offset_image).astype(dtype)

    # Add baseline:
    adu += baseline

    # Models pixel saturation:
    adu[adu > max_adu] = max_adu

    return adu


def _poisson(rnd: RandomState, lam, size):
    """Draw samples from a Poisson distribution using the given random state.

    Parameters
    ----------
    rnd : numpy.random.RandomState
        Random state instance to use for sampling.
    lam : float or numpy.typing.ArrayLike
        Expected number of events (lambda parameter). Must be >= 0.
    size : int or tuple of int
        Output shape.

    Returns
    -------
    samples : numpy.ndarray
        Array of Poisson-distributed samples with the given shape.
    """
    return rnd.poisson(lam=lam, size=size)


def _normal(rnd: RandomState, scale, size):
    """Draw samples from a zero-mean normal distribution using the given random state.

    Parameters
    ----------
    rnd : numpy.random.RandomState
        Random state instance to use for sampling.
    scale : float
        Standard deviation of the normal distribution.
    size : int or tuple of int
        Output shape.

    Returns
    -------
    samples : numpy.ndarray
        Array of normally-distributed samples with the given shape.
    """
    return rnd.normal(scale=scale, size=size)
