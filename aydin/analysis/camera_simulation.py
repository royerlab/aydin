# Function to add camera noise
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
    """
    Realistic noise simulation for scientific cameras.
    Adapted from Kyle Douglas blog: http://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/
    With additional ideas from: https://mwcraig.github.io/ccd-as-book/01-03-Construction-of-an-artificial-but-realistic-image.html


    Parameters
    ----------
    photons_per_second :
        Image representing the number of photons received on the camera per pixel per second
    exposure_time_s :
        Exposure time in seconds
    quantum_efficiency :
        Quantum efficiency - i.e. conversion factor between photons and electrons
    gain :
        Conversion factor between electrons and Analog-Digital-Units (ADU)
    gain_sigma :
        Unfortunately, not all gains are identical across the camera pixels,
        This parameter controls the spread of the gain.
    gain_column_sigma :
        And often each column of the detector has its own electronics that induce another source of column-dependent noise.
        This parameter controls the additional spread of the gain per column.
    offset_mean :
        Pixel  amplification offset noise mean value.
    offset_sigma :
        Pixel amplification offset noise sigma value.
    dark_current :
        Dark current, in electrons per pixel per second, which is the way manufacturers typically
        report it.
    dark_current_sigma :
        Unfortunately, the dark current is not identical for each and every pixel.
        This parameter controls the spread of the dark current.
    dark_current_column_sigma :
        And often each column of the detector has its own electronics that induce another source of column-dependent noise.
        This parameter controls the additional spread of the dark current per column.
    min_exposure_dark_current:
        Minimal exposure for the purpose of dark photons. The effects of the dark current do not completely vanish for very short exposures...
    num_hot_pixels:
        Number of hot pixels.
    num_cold_pixels:
        Number of cold pixels.
    probability_cosmic_ray:
        Probability per pixel per second that a cosmic ray will hit a camera pixel.
    bitdepth :
        Bit depth of each pixel fo the camera
    baseline :
        Baseline value for camera
    shot_rnd :
        Random state for each image (time dependent)
    camera_rnd :
            Random state for each camera (time indedependent, camera instance dependent)
    dtype :
        Integral dtype to return image in

    Returns
    -------

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

    # And often each column of the detector has its own electronics that induce another source of column-dependent noise:
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

    # And often each column of the detector has its own electronics that induce another source of column-dependent noise:
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

        y_max, x_max = electrons.shape
        ray_x = camera_rnd.randint(0, x_max, size=effective_num_of_rays)
        ray_y = camera_rnd.randint(0, y_max, size=effective_num_of_rays)
        dark_electrons[tuple([ray_y, ray_x])] += int(gain * 16)

    # Some pixels are hot:
    if num_hot_pixels > 0:
        y_max, x_max = dark_electrons.shape
        hot_x = camera_rnd.randint(0, x_max, size=num_hot_pixels)
        hot_y = camera_rnd.randint(0, y_max, size=num_hot_pixels)
        dark_electrons[tuple([hot_y, hot_x])] *= min(16, 2 ** (bitdepth - 2))

    # Some pixels are cold:
    if num_cold_pixels > 0:
        y_max, x_max = electrons.shape
        cold_x = camera_rnd.randint(0, x_max, size=num_cold_pixels)
        cold_y = camera_rnd.randint(0, y_max, size=num_cold_pixels)
        electrons[tuple([cold_y, cold_x])] /= min(16, 2 ** (bitdepth - 2))

    # Add dark current
    all_electrons = dark_electrons + electrons

    # max ADU:
    max_adu = numpy.int(2 ** bitdepth - 1)

    # Convert to discrete numbers (ADU):
    adu = (all_electrons * gain_image + offset_image).astype(dtype)

    # Add baseline:
    adu += baseline

    # Models pixel saturation:
    adu[adu > max_adu] = max_adu

    return adu


def _poisson(rnd: RandomState, lam, size):
    return rnd.poisson(lam=lam, size=size)


def _normal(rnd: RandomState, scale, size):
    return rnd.normal(scale=scale, size=size)
