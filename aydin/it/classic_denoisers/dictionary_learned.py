import math
from typing import Optional
import numpy

from aydin.it.classic_denoisers.dictionary_fixed import denoise_dictionary_fixed
from aydin.util.crop.rep_crop import representative_crop
from aydin.util.dictionary.dictionary import learn_dictionary
from aydin.util.j_invariance.j_invariant_classic import calibrate_denoiser_classic
from aydin.util.patch_size.patch_size import default_patch_size


def calibrate_denoise_dictionary_learned(
    image,
    patch_size: int = None,
    try_omp: bool = True,
    try_lasso_lars: bool = False,
    try_lasso_cd: bool = False,
    try_lars: bool = False,
    try_threshold: bool = False,
    max_patches: Optional[int] = int(1e6),
    dictionary_size: Optional[int] = None,
    over_completeness: float = 3,
    max_dictionary_size: int = 256,
    try_kmeans: bool = True,
    try_pca: bool = True,
    try_ica: bool = False,
    try_sdl: bool = False,
    num_sparsity_values_to_try: int = 6,
    num_iterations: int = 1024,
    batch_size: int = 3,
    alpha: int = 1,
    do_cleanup_dictionary: bool = True,
    do_denoise_dictionary: bool = False,
    crop_size_in_voxels: Optional[int] = None,
    display_dictionary: bool = False,
    display_images: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates the dictionary-based denoiser for the given image by finding the
    best learned dictionary of patches and returns the optimal parameters.

    Parameters
    ----------
    image : ArrayLike
        Image to calibrate denoiser for.

    patch_size : int
        Patch size. If None it is automatically adjusted
        to teh number of dimensions of the image to
        ensure a reasonable computational effort.
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

    max_patches: Optional[int]
        Max number of patches to extract for dictionary
        learning. If None there is no limit.
        (advanced)

    dictionary_size: int
        Dictionary size in 'atoms'. If None the dictionary
        size is inferred from the over_completeness
        parameter type.
        (advanced)

    over_completeness: float
        Given a certain patch size p and image dimension
        n, a complete basis has p^n elements, the over
        completeness factor determines the size of
        the dictionary relative to that by the formula:
        ox*p^n.
        (advanced)

    max_dictionary_size: int
        Independently of any other parameter, we limit the
        size of the dictionary to this provided number.
        (advanced)

    try_kmeans: bool
        Whether to compute a kmeans based dictionary
        and used it as one of possible dictionaries
        during calibration.

    try_pca: bool
        Whether to compute a PCA based dictionary and
        used it as one of possible dictionaries during
        calibration.

    try_ica: bool
        Whether to compute an ICA (Independent
        Component Analysis) based dictionary and
        used it as one of possible dictionaries
        during calibration.

    try_sdl: bool
        Whether to compute a SDL (Sparse Dictionary
        Learning) based dictionary and used it as one of
        possible dictionaries during calibration.

    num_sparsity_values_to_try: int
        Maximum number of sparsity values to try during calibration
        (advanced)

    num_iterations: int
        Number of iterations for learning dictionary.
        (advanced)

    batch_size: int
        Size of batches during batched dictionary
        learning.
        (advanced)

    alpha: int
        Sparsity prior strength.
        (advanced)

    do_cleanup_dictionary: bool
        Removes dictionary entries that are likely pure
        noise or have impulses or very high-frequencies
        or checkerboard patterns that are unlikely
        needed to reconstruct the true signal.
        (advanced)

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate
        denoiser.
        (advanced)

    display_dictionary: bool
        If True displays dictionary with napari --
        for debug purposes.

    display_images: bool
        When True the denoised images encountered
        during optimisation are shown.

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

    # algorithms to try for generation of dictionaries:
    algorithms = []
    if try_kmeans:
        algorithms.append('kmeans')
    if try_pca:
        algorithms.append('pca')
    if try_ica:
        algorithms.append('ica')
    if try_sdl:
        algorithms.append('sdl')

    # dictionaries to try:
    dictionaries = {}

    for algorithm in algorithms:
        # learn dictionary:
        dictionary = learn_dictionary(
            image,
            patch_size=patch_size,
            max_patches=max_patches,
            dictionary_size=dictionary_size,
            over_completeness=over_completeness,
            max_dictionary_size=max_dictionary_size,
            algorithm=algorithm,
            num_iterations=num_iterations,
            batch_size=batch_size,
            alpha=alpha,
            cleanup_dictionary=do_cleanup_dictionary,
            denoise_dictionary=do_denoise_dictionary,
            display_dictionary=display_dictionary,
            **other_fixed_parameters,
        )
        dictionaries[algorithm] = dictionary

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
        'sparsity': [1, 2, 3, 4, 8, 16][:num_sparsity_values_to_try],
        'gamma': [0.001],
        'coding_mode': coding_modes,
        'dictlearn_algorithm': algorithms
        # 'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'
    }

    # Partial function:
    def _denoise_dictionary(image, dictlearn_algorithm, *args, **kwargs):
        return denoise_dictionary_learned(
            image,
            *args,
            dictionary=dictionaries[dictlearn_algorithm],
            **other_fixed_parameters,
            **kwargs,
        )

    # Calibrate denoiser:
    best_parameters = (
        calibrate_denoiser_classic(
            crop,
            _denoise_dictionary,
            denoise_parameters=parameter_ranges,
            display_images=display_images,
        )
        | other_fixed_parameters
    )

    # Memory needed:
    memory_needed = 2 * image.nbytes + 6 * image.nbytes * math.prod(patch_size)

    return denoise_dictionary_learned, best_parameters, memory_needed


def denoise_dictionary_learned(*args, **kwargs):
    """
    Denoises the given image using sparse-coding over a fixed
    dictionary of nD image patches. The dictionary learning and
    patch sparse coding uses scikit-learn's Batch-OMP implementation.

    Parameters
    ----------
    args
    kwargs

    Returns
    -------
    denoised image
    """
    return denoise_dictionary_fixed(*args, **kwargs)
