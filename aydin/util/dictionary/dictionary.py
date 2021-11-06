from math import prod
from typing import Optional, Tuple
import numpy
from numpy.linalg import norm
from scipy.fft import idstn, idctn
from scipy.ndimage import convolve
from scipy.ndimage import median_filter, gaussian_filter
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.decomposition import MiniBatchDictionaryLearning as DictLearning


from aydin.util.patch_size.patch_size import default_patch_size
from aydin.util.patch_transform.patch_transform import extract_patches_nd
from aydin.util.log.log import lsection


def learn_dictionary(
    image,
    patch_size: int = 7,
    max_patches: Optional[int] = int(1e6),
    dictionary_size: Optional[int] = None,
    over_completeness: float = 3,
    max_dictionary_size: int = 256,
    algorithm: str = 'kmeans',
    num_iterations: int = 1024,
    batch_size: int = 3,
    alpha: int = 1,
    cleanup_dictionary: bool = True,
    denoise_dictionary: bool = False,
    display_dictionary: bool = False,
    **kwargs,
):
    """
    Learns a dictionary from an image using sparse coding over an
    over-complete learned dictionary. The dictionary learning uses
    scikit-learn's Batch-OMP implementation.

    Parameters
    ----------
    image: ArrayLike
        nD image to be denoised

    patch_size: int
        Patch size

    max_patches: Optional[int]
        Max number of patches to extract for dictionary learning.
        If None there is no limit.

    dictionary_size: int
        Dictionary size in 'atoms'. If None the dictionary size is inferred from
        the over_completeness parameter

    over_completeness: float
        Given a certain patch size p and image dimension n, a complete basis
        has p^n elements, the over completeness factor oc determintes the size
        of the dictionary relative to that by the formula: ox*p^n

    max_dictionary_size: int
        Independently of any other parameter, we limit the
        size of the dictionary to this provided number.

    algorithm: str
        Algorithm used to compute the dictionary.
        Can be: 'sdl' (sparse dictionary learning),
         'ica' (independent component analysis),
         or 'kmeans' or 'pca'

    num_iterations: int
        Number of iterations for learning dictionary

    batch_size: int
        Size of batches during batched dictionary learning

    alpha: int
        Sparsity prior strength.

    cleanup_dictionary: bool
        Removes dictionary entries that are likely pure noise or have impulses
        or very high-frequencies or checkerboard patterns that are unlikely
        needed to reconstruct the true signal.

    denoise_dictionary: bool
        Applies denosing to the dictionary atoms. Can be 'median' or 'gaussian'.

    display_dictionary: bool
        If True displays dictionary with napari -- for debug purposes.

    Returns
    -------
    Learned dictionary as a list of patches of shape: (n,)+path_size
    where n is the number of patches in the dictionary
    """

    # Normalise patch size:
    patch_size = default_patch_size(image, patch_size, odd=False)

    # Dictionary size can be computed automatically, by default we have an overcomplete factor of 4:
    dictionary_size = (
        int(over_completeness * prod(patch_size))
        if dictionary_size is None
        else dictionary_size
    )

    # limit dictionary size to max:
    dictionary_size = min(max_dictionary_size, dictionary_size)

    # Adjusting dictionary size on the basis of the algorithm chosen:
    if algorithm == 'ica' or algorithm == 'pca':
        dictionary_size = min(dictionary_size, int(numpy.prod(patch_size)))

    # extract normalised patches for learning dictionary:
    patches = extract_normalised_vectorised_patches(
        image, patch_size=patch_size, max_patches=max_patches, normalise_stds=True
    )

    with lsection(
        f"Learning dictionary of {dictionary_size} atoms from {len(patches)} patches using algorithm '{algorithm}'."
    ):

        if algorithm == 'sdl':
            learner = DictLearning(
                n_components=dictionary_size,
                alpha=alpha,
                n_jobs=-1,
                n_iter=num_iterations,
                batch_size=batch_size,
            )
            learner.fit(patches)
            atoms = learner.components_
        elif algorithm == 'ica':
            learner = FastICA(n_components=dictionary_size, max_iter=num_iterations)
            learner.fit(patches)
            atoms = learner.components_
        elif algorithm == 'kmeans':
            learner = KMeans(
                n_clusters=dictionary_size,
                max_iter=num_iterations,
                n_init=8,
                batch_size=2 * dictionary_size,
                reassignment_ratio=0.01,
            )
            learner.fit(patches)
            atoms = learner.cluster_centers_
        elif algorithm == 'pca':
            learner = PCA(n_components=dictionary_size)
            learner.fit(patches)
            atoms = learner.components_
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Reshape to patches:
        atoms = atoms.reshape(len(atoms), *patch_size)

        # dictionary cleanup:
        if cleanup_dictionary:
            atoms = dictionary_cleanup(atoms)

        if denoise_dictionary == 'median':
            atoms = numpy.stack(median_filter(i, size=3) for i in atoms)
        elif denoise_dictionary == 'gaussian':
            atoms = numpy.stack(gaussian_filter(i, sigma=0.66) for i in atoms)

        if display_dictionary:
            import napari

            with napari.gui_qt():
                viewer = napari.Viewer()
                viewer.add_image(
                    atoms.reshape(len(atoms), *patch_size), name='dictionary'
                )

    return atoms


def extract_normalised_vectorised_patches(
    image,
    patch_size: Tuple[int],
    max_patches: Optional[int],
    normalise_means: bool = True,
    normalise_stds: bool = True,
    output_norm_values: bool = False,
):
    # extracts patches:
    patches = extract_patches_nd(image, patch_size=patch_size, max_patches=max_patches)

    # vectorise patches as 1D vector:
    patches = patches.reshape(patches.shape[0], -1)

    # Normalise means:
    if normalise_means:
        patches_means = numpy.mean(patches, axis=0)
        patches -= patches_means
    elif output_norm_values:
        patches_means = numpy.zeros(len(patches))[..., numpy.newaxis]

    # Normalise stds:
    if normalise_stds:
        patches_stds = numpy.std(patches, axis=0)
        # We cannot divide by zero:
        patches_stds[patches_stds == 0] = 1
        patches /= patches_stds
    elif output_norm_values:
        patches_stds = numpy.ones(len(patches))[..., numpy.newaxis]

    if output_norm_values:
        return patches, patches_means, patches_stds
    else:
        return patches


def fixed_dictionary(
    image,
    patch_size: int = 7,
    dictionaries: str = 'dst+dct',
    max_freq: float = 0.66,
    display_dictionary: bool = False,
    **kwargs,
):
    """
    Returns a fixed dictionary consisting in the concatenations different
    fixed dictionaries.

    Parameters
    ----------
    image: ArrayLike
        nD image to be denoised

    patch_size: int
        Patch size

    dictionaries: str
        Fixed dictionaries to be included. Can be: 'dct', 'dst'

    max_freq: float
        Maximal allowed frequency for dct and dst

    display_dictionary: bool
        If True displays dictionary with napari -- for debug purposes.

    Returns
    -------
    Learned dictionary as a list of patches of shape: (n,)+path_size
    where n is the number of patches in the dictionary
    """

    # Normalise patch size:
    patch_size = default_patch_size(image, patch_size, odd=True)

    atoms = []

    if 'dct' in dictionaries or 'dst' in dictionaries:
        dctcoefs = numpy.zeros(shape=patch_size)

        for index, _ in numpy.ndenumerate(dctcoefs):

            freqs = numpy.array([u / s for u, s in zip(index, patch_size)])
            freq = norm(freqs)

            if freq > max_freq:
                continue

            dctcoefs[...] = 0
            dctcoefs[index] = 1

            if 'dct' in dictionaries:
                kernel = idctn(dctcoefs, norm="ortho")
                kernel = kernel.astype(numpy.float32)
                atoms.append(kernel)

            if 'dst' in dictionaries:
                kernel = idstn(dctcoefs, norm="ortho")
                kernel = kernel.astype(numpy.float32)
                atoms.append(kernel)

    # turn list into numpy array:
    atoms = numpy.stack(atoms)

    # Display if needed:
    if display_dictionary:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(atoms.reshape(len(atoms), *patch_size), name='dct_atoms')

    return atoms


def dictionary_cleanup(
    patches,
    filters: str = 'impulse+fractured+lipschitz',
    truncate: float = 0.10,
    display: bool = False,
):
    """
    Performs different kinds of patch filtering to remove 'bad patches' in a
    dictionary.

    Parameters
    ----------
    patches : ArrayLike
        Patches to filter

    filters: str
        Filters to remove.

    truncate: float
        Percentage of the worst patches to remove.

    display: bool
        Display patches

    Returns
    -------
    Patches to remove.

    """

    # First we make a list:
    patches = list([patch for patch in patches])

    if 'impulse' in filters:
        filtered_patches = list(filter(lambda p: not _is_impulse(p), patches))

    if 'fractured' in filters:
        filtered_patches = sorted(filtered_patches, key=lambda p: _fracture_measure(p))
        filtered_patches = filtered_patches[: -int(len(filtered_patches) * truncate)]

    if 'lipschitz' in filters:
        filtered_patches = sorted(filtered_patches, key=lambda p: _lipschitz_error(p))
        filtered_patches = filtered_patches[: -int(len(filtered_patches) * truncate)]

    if display:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(numpy.stack(patches), name='patches')
            viewer.add_image(numpy.stack(filtered_patches), name='filtered_patches')

    return numpy.stack(filtered_patches)


def _is_impulse(patch):
    if patch.min() < patch.max():
        patch = patch.copy()
        patch -= patch.min()
        patch /= patch.max()

    number_of_positive = (patch[patch > 0.5]).size
    number_of_negative = (patch[patch < 0.5]).size

    return number_of_positive == 1 or number_of_negative == 1


def _fracture_measure(patch):
    # Footprint:
    footprint_shape = (3,) * patch.ndim
    footprint_a = numpy.zeros(footprint_shape, dtype=numpy.float32)

    for index in numpy.ndindex(footprint_shape):
        index_sum = sum(index) + patch.ndim
        footprint_a[tuple(slice(i, i + 1) for i in index)] = index_sum % 2

    footprint_b = 1 - footprint_a

    footprint_a /= numpy.sum(footprint_a)
    footprint_b /= numpy.sum(footprint_b)

    value = numpy.max(
        numpy.abs(
            convolve(patch, weights=footprint_a) - convolve(patch, weights=footprint_b)
        )
    )

    return value


def _lipschitz_error(patch, lipschitz: float = 0.2):
    # we compute the error map:
    median = median_filter(patch, size=3)
    error = median.copy()
    error -= patch
    numpy.abs(error, out=error)
    numpy.maximum(error, lipschitz, out=error)
    error -= lipschitz

    _lipschitz_error = numpy.amax(error)

    return _lipschitz_error
