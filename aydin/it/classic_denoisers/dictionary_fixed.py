import math
from typing import Optional

import numpy
from sklearn.decomposition import SparseCoder

from aydin.util.crop.rep_crop import representative_crop
from aydin.util.dictionary.dictionary import (
    fixed_dictionary,
    extract_normalised_vectorised_patches,
)
from aydin.util.j_invariance.j_invariant_classic import calibrate_denoiser_classic
from aydin.util.j_invariance.j_invariant_smart import calibrate_denoiser_smart
from aydin.util.log.log import lsection, lprint
from aydin.util.patch_size.patch_size import default_patch_size
from aydin.util.patch_transform.patch_transform import reconstruct_from_nd_patches


def calibrate_denoise_dictionary_fixed(
    image,
    patch_size: int = None,
    try_omp: bool = True,
    try_lasso_lars: bool = False,
    try_lasso_cd: bool = False,
    try_lars: bool = False,
    try_threshold: bool = False,
    num_sparsity_values_to_try: int = 6,
    dictionaries: str = 'dct',
    crop_size_in_voxels: Optional[int] = None,
    max_num_evaluations: int = 256,
    display_dictionary: bool = False,
    display_images: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates the dictionary-based denoiser for the given image and returns the
    optimal parameters obtained using the N2S loss.

    Parameters
    ----------
    image : ArrayLike
        Image to calibrate denoiser for.

    patch_size : int
        Patch size. Common parameter to both 'learned',
        or 'fixed' dictionary types.
        (advanced)

    try_omp: bool
        Whether OMP should be tried as a sparse coding
        algorithm during calibration.

    try_lasso_lars: bool
        Whether LASSO-LARS should be tried as a sparse
        coding algorithm during calibration.

    try_lasso_cd: bool
        Whether LASSO-CD should be tried as a sparse
        coding algorithm during calibration.

    try_lars: bool
        Whether LARS should be tried as a sparse coding
        algorithm during calibration.

    try_threshold: bool
        Whether 'threshold'' should be tried as a sparse
        coding algorithm during calibration.

    num_sparsity_values_to_try: int
        Maximum number of sparsity values to try during calibration
        (advanced)

    dictionaries: str
        Fixed dictionaries to be included. Can be: 'dct',
        'dst'.

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate
        denoiser.
        (advanced)

    max_num_evaluations: int
        Maximum number of evaluations for finding the
        optimal parameters.
        (advanced)

    display_dictionary: bool
        If True displays dictionary with napari -- for
        debug purposes.

    display_images: bool
        When True the denoised images encountered during
        optimisation are shown.

    other_fixed_parameters: dict
        Any other fixed parameters


    Returns
    -------
    Denoising function, dictionary containing optimal parameters,
    and free memory needed in bytes for computation.
    """
    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    # Normalise patch size:
    patch_size = default_patch_size(image, patch_size, odd=True)

    # obtain representative crop, to speed things up...
    crop = representative_crop(image, crop_size=crop_size_in_voxels)

    # Partial function:
    def _denoise_dictionary(
        image, max_freq: float = 0.5, coding_mode: str = 'omp', **parameters
    ):
        dictionary = fixed_dictionary(
            image, patch_size=patch_size, dictionaries=dictionaries, max_freq=max_freq
        )
        denoised_image = denoise_dictionary_fixed(
            image, dictionary=dictionary, coding_mode=coding_mode, **parameters
        )
        return denoised_image

    # coding modes to try:
    coding_modes = []
    if try_omp:
        coding_modes.append('omp')
    if try_lasso_lars:
        coding_modes.append('lasso_lars')
    if try_lasso_cd:
        coding_modes.append('lasso_cd')
    if try_lars:
        coding_modes.append('lars')
    if try_threshold:
        coding_modes.append('threshold')

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {
        'max_freq': (0.01, 1.3),
        # numpy.arange(0.05, 1.4, 0.05),  # (0.01, 1.3),
        'coding_mode': coding_modes,
        # 'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'
    }

    # Calibrate denoiser:
    best_parameters = calibrate_denoiser_smart(
        crop,
        _denoise_dictionary,
        denoise_parameters=parameter_ranges,
        max_num_evaluations=max_num_evaluations,
    )
    lprint(f"Best parameters: {best_parameters}")

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {
        'sparsity': [1, 2, 3, 4, 8, 16][:num_sparsity_values_to_try],
        # 'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'
    }

    # Calibrate denoiser:
    best_parameters = (
        calibrate_denoiser_classic(
            crop,
            _denoise_dictionary,
            denoise_parameters=parameter_ranges,
            other_fixed_parameters=best_parameters | other_fixed_parameters,
            display_images=display_images,
        )
        | best_parameters
        | other_fixed_parameters
    )

    # Cleaning up a bit:
    best_parameters.pop('other_fixed_parameters')

    lprint(f"Final best parameters: {best_parameters}")

    # we need to replace the max freq argument with the actual dictionary
    # because that's what our client facing denoise function expects:
    max_freq = best_parameters.pop('max_freq')

    # Dictionary to use based on fixed and best parameters:
    dictionary = fixed_dictionary(
        image, patch_size=patch_size, dictionaries=dictionaries, max_freq=max_freq
    )

    best_parameters = best_parameters | {'dictionary': dictionary}

    if display_dictionary:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(
                dictionary.reshape(len(dictionary), *patch_size), name='dictionary'
            )

    # Memory needed:
    memory_needed = 2 * image.nbytes + 6 * image.nbytes * math.prod(patch_size)

    return denoise_dictionary_fixed, best_parameters, memory_needed


def denoise_dictionary_fixed(
    image,
    dictionary=None,
    coding_mode: str = 'omp',
    sparsity: int = 1,
    gamma: float = 0.001,
    multi_core: bool = True,
    **kwargs,
):
    """
    Denoises the given image using sparse-coding over a fixed
    dictionary of nD image patches. The dictionary learning and
    patch sparse coding uses scikit-learn's Batch-OMP implementation.

    Parameters
    ----------
    image: ArrayLike
        nD image to be denoised

    dictionary: ArrayLike
        Dictionary to use for denosing image via sparse coding.
        By default (None) a fixed dictionary is used.

    coding_mode: str
        Type of sparse coding, can be:  'lasso_lars', 'lasso_cd', 'lars', 'omp',
        or 'threshold'

    sparsity: int
        How many atoms are used to represent each patch after denoising.

    gamma: float
        How much the periphery of teh patches contributes to the final denoised
        image. Larger gamma means that we keep more of the central pixels of the
        patches, smaller values lead to a more uniform contribution.
        A value of 1 corresponds to the default blackman window.

    multi_core: bool
        By default we use as many cores as possible, in some cases, for small
        (test) images, it might be faster to run on a single core instead of
        starting the whole parallelization machinery.

    Returns
    -------
    Denoised image
    """

    # Convert image to float if needed:
    image = image.astype(dtype=numpy.float32, copy=False)

    if dictionary is None:
        # learn dictionary with all defaults:
        dictionary = fixed_dictionary(image)

    # we can infer patch shape from dictionary:
    patch_size = dictionary.shape[1:]

    with lsection(f"Denoise image of shape {image.shape} and dtype {image.dtype}"):
        # vectorise dictionary:
        vectorised_dictionary = dictionary.reshape(len(dictionary), -1)

        # setup sparse coder:
        coder = SparseCoder(
            vectorised_dictionary,
            transform_algorithm=coding_mode,
            transform_n_nonzero_coefs=sparsity,
            n_jobs=-1 if multi_core else 1,
        )

        # First we extract _all_ patches from the image, without any normalisation:
        with lsection("Extract all patches from image..."):
            patches, patch_means, _ = extract_normalised_vectorised_patches(
                image,
                patch_size=patch_size,
                max_patches=None,
                normalise_means=True,
                normalise_stds=False,
                output_norm_values=True,
            )

        with lsection("Obtain sparse codes for each patch..."):
            code = coder.transform(patches)

        with lsection("Reconstruct patches from codes..."):
            denoised_patches = numpy.dot(code, vectorised_dictionary)
            # Add back means:
            denoised_patches += patch_means

        with lsection("Reshape to patches..."):
            denoised_patches = denoised_patches.reshape(len(patches), *patch_size)

        with lsection("Reconstructing image from patches..."):
            # Reconstructs image from denoised patches:
            denoised_image = reconstruct_from_nd_patches(
                patches=denoised_patches, image_shape=image.shape, gamma=gamma
            )

    return denoised_image
