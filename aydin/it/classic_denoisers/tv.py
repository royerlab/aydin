from functools import partial
from typing import Optional, Tuple, List

import numpy
from numba import jit
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_tv_bregman
from skimage.restoration._denoise import _denoise_tv_chambolle_nd

from aydin.it.classic_denoisers import _defaults
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.j_invariance.j_invariance import calibrate_denoiser


def calibrate_denoise_tv(
    image: ArrayLike,
    enable_mixing: bool = True,
    crop_size_in_voxels: Optional[int] = _defaults.default_crop_size_normal.value,
    optimiser: str = _defaults.default_optimiser.value,
    max_num_evaluations: int = _defaults.default_max_evals_high.value,
    blind_spots: Optional[List[Tuple[int]]] = _defaults.default_blind_spots.value,
    jinv_interpolation_mode: str = _defaults.default_jinv_interpolation_mode.value,
    display_images: bool = False,
    display_crop: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates the Total Variation (TV) denoiser for the given image and
    returns the optimal parameters obtained using the N2S loss.

    Note: we use the scikt-image implementation of TV denoising.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate TV denoiser for.

    enable_mixing: bool
        TV denoising tends to return very non-natural looking piece-wise constant
        images. To mitigate this  we give the option to mix a bit of the original
        image denoised with a Gaussian filter. The sigma for the Gaussian filter
        is also calibrated. Note: In some cases it might be simply better to not
        use the TV denoised image and instead just use the
        Gaussian-filter-denoised image, check the "beta" parameter during
        optimisation, if that number is very close to 1.0 then you know you
        should probably use a Butterworth or Gaussian denoiser.

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate denoiser.
        Increase this number by factors of two if denoising quality is
        unsatisfactory -- this can be important for very noisy images.
        Values to try are: 65000, 128000, 256000, 320000.
        We do not recommend values higher than 512000.

    optimiser: str
        Optimiser to use for finding the best denoising
        parameters. Can be: 'smart' (default), or 'fast' for a mix of SHGO
        followed by L-BFGS-B.
        (advanced)

    max_num_evaluations: int
        Maximum number of evaluations for finding the optimal parameters.
        Increase this number by factors of two if denoising quality is
        unsatisfactory.

    blind_spots: bool
        List of voxel coordinates (relative to receptive field center) to
        be included in the blind-spot. For example, you can give a list of
        3 tuples: [(0,0,0), (0,1,0), (0,-1,0)] to extend the blind spot
        to cover voxels of relative coordinates: (0,0,0),(0,1,0), and (0,-1,0)
        (advanced) (hidden)

    jinv_interpolation_mode: str
        J-invariance interpolation mode for masking. Can be: 'median' or
        'gaussian'.
        (advanced)

    display_images: bool
        When True the denoised images encountered during optimisation are shown.
        (advanced) (hidden)

    display_crop: bool
        Displays crop, for debugging purposes...
        (advanced) (hidden)

    other_fixed_parameters: dict
        Any other fixed parameters

    Returns
    -------
    Denoising function, dictionary containing optimal parameters,
    and free memory needed in bytes for computation.

    """
    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # obtain representative crop, to speed things up...
    crop = representative_crop(
        image, crop_size=crop_size_in_voxels, display_crop=display_crop
    )

    # Sigma spatial range:
    range = max(1e-10, numpy.max(image) - numpy.min(image))
    # weight_range = np.arange(0.01, 1.5*std, 0.05*std) ** 1.5
    weight_range = (0.01 * range, 1.5 * range)

    # Algorithms:
    algorithms = ['bregman', 'chambolle'] if image.ndim <= 2 else ['chambolle']
    # Note: scikitimage implementation of bregman does not support more than 2D

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {
        'weight': weight_range,
        'isotropic': [False, True],
        'algorithm': algorithms,
    }

    if enable_mixing:
        # alpha beta range:
        alpha_range = (0.0, 1.0)
        beta_range = (0.0, 1.0)
        sigma_range = (0.0, 2.0)

        # Add parameters to be optimised:
        parameter_ranges |= {
            'alpha': alpha_range,
            'beta': beta_range,
            'sigma': sigma_range,
        }

    # Partial function:
    _denoise_tv = partial(denoise_tv, **other_fixed_parameters)

    # Calibrate denoiser
    best_parameters = (
        calibrate_denoiser(
            crop,
            _denoise_tv,
            mode=optimiser,
            denoise_parameters=parameter_ranges,
            interpolation_mode=jinv_interpolation_mode,
            max_num_evaluations=max_num_evaluations,
            blind_spots=blind_spots,
            display_images=display_images,
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = image.nbytes * 2  # gradient image

    return denoise_tv, best_parameters, memory_needed


def denoise_tv(
    image: ArrayLike,
    algorithm: str = 'bregman',
    weight: float = 1,
    alpha: float = 1.0,
    beta: float = 0.0,
    sigma: float = 0.0,
    **kwargs,
):
    """
    Denoises the given image using either scikit-image
    implementation of Bregman or Chambolle <a
    href="https://en.wikipedia.org/wiki/Total_variation_denoising">Total
    Variation (TV) denoising</a>. We attempt to rescale the weight parameter to
    obtain similar results between Bregman and Chambolle for the
    same weight.
    \n\n
    Note: Because of limitations of the current scikit-image
    implementation, if an image with more than 2 dimensions
    is passed, we use the Chambolle implementation as it
    supports nD images...

    Parameters
    ----------
    image: ArrayLike
        Image to denoise

    algorithm: str
        Algorithm to apply: either 'bregman' or 'chambolle'

    weight: float
        Weight to balance data term versus prior term. Larger values correspond
        to a stronger prior during optimisation, and thus more aggressive
        denoising

    alpha: float
        TV denoising tends to return images that are dimmer, we have the option
        of pushing the brightness here..

    beta: float
        TV denoising tends to return very non-natural looking images, here we give
        the option to 'mix-in' back a bit of the original image denoised with a
        gaussian filter.

    sigma: float
        Sigma for gaussian filtered image that is mixed with the TV denoised image.

    kwargs
        Any other parameters to be passed to scikit-image implementations

    Returns
    -------
    Denoised image
    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    if algorithm == 'bregman' and image.ndim <= 2:
        denoised = denoise_tv_bregman(image, weight=weight * 10, **kwargs)

    elif algorithm == 'chambolle' or image.ndim > 2:
        if 'isotropic' in kwargs:
            kwargs.pop('isotropic')
        denoised = _denoise_tv_chambolle_nd(image, weight=weight, **kwargs)

    if (alpha != 1.0 or beta > 1e-3) and sigma != 0.0:
        # image = median_filter(image, size=3)
        image = gaussian_filter(image, sigma=sigma)
        denoised = _mixin(denoised, image, alpha, beta)

    return denoised


@jit(nopython=True, parallel=True)
def _mixin(image_a: ArrayLike, image_b: ArrayLike, alpha: float, beta: float):
    return alpha * image_a + beta * image_b
