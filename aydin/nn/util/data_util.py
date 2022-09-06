import numpy
from deprecated import deprecated
from scipy.stats import entropy


def random_sample_patches(
    image, patch_size: int, nb_patches_per_image: int, adoption_rate: float = 0.5
):
    """
    This functions returns list of slice objects that crops a part of the image
    which we call patch. Also sorts the patches, and makes sure only patches with
    higher entropy in the intensity histogram are selected.

    Parameters
    ----------
    image : numpy.ArrayLike
        This function assumes the axis order BXY(Z)C.
    patch_size : int
    nb_patches_per_image : int
    adoption_rate : float

    Returns
    -------

    """
    list_of_slice_objects = []
    print(image.shape, patch_size, nb_patches_per_image, adoption_rate)

    # Calculate total number of possible patches for given image and patch_size
    possible_positions = numpy.asarray(image.shape[1:-1]) - patch_size + 1
    nb_possible_patches_per_image = numpy.prod(possible_positions)

    # Validate nb_patches_per_image, adoption_rate combination is valid, if not generate all possible patches
    nb_patches_per_image = min(
        int(nb_patches_per_image / adoption_rate), nb_possible_patches_per_image
    )

    for b in range(image.shape[0]):  # b is a single element across batch dimension
        entropies = []
        slice_objects_for_current_b = []

        # Generate patches and entropy values
        while len(slice_objects_for_current_b) < nb_patches_per_image:
            indices_for_current_patch = (
                [b] + [int(numpy.random.choice(s, 1)) for s in possible_positions] + [0]
            )
            slicing_for_current_patch = tuple(
                slice(
                    x,
                    x + (1 if idx == 0 or idx == len(image.shape) - 1 else patch_size),
                    1,
                )
                for idx, x in enumerate(indices_for_current_patch)
            )
            slice_objects_for_current_b.append(slicing_for_current_patch)

            current_patch = image[slicing_for_current_patch]

            # Calculate histogram and entropy
            hist, _ = numpy.histogram(
                current_patch, range=(0, 1), bins=255, density=True
            )
            entropies.append(entropy(hist))

        # Sort patches

        # Filter patches according to adoption_rate

        # Append the new patch slices to list_of_slice_objects

    response = numpy.vstack(list_of_slice_objects)
    return response


@deprecated(
    version='0.1.14', reason="You should use random_sample_patches function instead"
)
def legacy_random_sample_patches(input_img, patch_size, num_patch, adoption_rate=0.5):
    """
    This function outputs a list of slices that crops a part of the input_img (i.e. patch).
    Only patches with higher entropy in their intensity histogram are selected.

    Parameters
    ----------
    input_img
        input image that will be sampled with patches
    patch_size
        patch_size
    num_patch
        number of patches to be output
    adoption_rate
        The % of patches selected from the original population of patches

    Returns
    -------
    patch_slice
    """
    if type(patch_size) == int:
        if len(input_img.shape) == 4:
            patch_size = (patch_size, patch_size)
        if len(input_img.shape) == 5:
            patch_size = (patch_size, patch_size, patch_size)

    img_dim = input_img.shape
    patchs_per_img = numpy.ceil(
        numpy.ceil(num_patch / img_dim[0]) / adoption_rate
    ).astype(int)
    coordinates = numpy.asarray(img_dim[1:-1]) - numpy.asarray(patch_size)
    patchs_per_img = min(patchs_per_img, numpy.prod(coordinates + 1))
    hist_ind_all = []
    for k in range(img_dim[0]):
        ind_past = []
        hist_batch = []
        while len(hist_batch) < patchs_per_img:
            # Randomly choose coordinates from an image.
            ind = numpy.hstack(
                [k]
                + [
                    numpy.random.choice(
                        1 if coordinates[i] == 0 else coordinates[i], 1, replace=True
                    ).astype(int)
                    for i in range(coordinates.size)
                ]
            )
            # Check if the new patch is too close to the existing patches.
            if ind_past:
                if abs(ind_past - ind).max() <= coordinates.min() // 20:
                    continue
            ind_past.append(ind)
            # Extract image patch from the input image.
            if len(patch_size) == 2:
                img0 = input_img[
                    ind[0],
                    ind[1] : ind[1] + patch_size[0],
                    ind[2] : ind[2] + patch_size[1],
                    0,
                ]
            elif len(patch_size) == 3:
                img0 = input_img[
                    ind[0],
                    ind[1] : ind[1] + patch_size[0],
                    ind[2] : ind[2] + patch_size[1],
                    ind[3] : ind[3] + patch_size[2],
                    0,
                ]
            else:
                raise ValueError('Only 2D or 3D patches are applicable.')
            # Calculate histogram and entropy
            hist, _ = numpy.histogram(img0, range=(0, 1), bins=255, density=True)
            hist_batch.append(entropy(hist))
        # Create a table with entropy and indices of each patch.
        hist_ind = numpy.hstack((numpy.vstack(hist_batch), ind_past))
        # Sort by entropy.
        hist_ind = hist_ind[(-hist_ind[:, 0]).argsort()]
        # Only leave the highest `adoption_rate` of patches.
        hist_ind = hist_ind[: max(int(hist_ind.shape[0] * adoption_rate), 1), ...]
        hist_ind_all.append(hist_ind)
    hist_ind_all = numpy.vstack(hist_ind_all)
    hist_ind_all = hist_ind_all[(-hist_ind_all[:, 0]).argsort()]
    hist_ind_all = hist_ind_all[: int(num_patch), 1:].astype(int)

    # Create a slice list
    patch_slices = []
    for ind in hist_ind_all:
        slice_list = (
            [slice(ind[0], ind[0] + 1, 1)]
            + [
                slice(ind[i + 1], ind[i + 1] + patch_size[i], 1)
                for i in range(len(patch_size))
            ]
            + [slice(0, 1, 1)]
        )
        patch_slices.append(tuple(slice_list))

    # Return a list of slice
    return patch_slices
