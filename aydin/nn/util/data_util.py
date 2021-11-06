import numpy
from scipy.stats import entropy


def random_sample_patches(input_img, patch_size, num_patch, adoption_rate=0.5):
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
    patch_slice = []
    for ind in hist_ind_all:
        slice_list = (
            [slice(ind[0], ind[0] + 1, 1)]
            + [
                slice(ind[i + 1], ind[i + 1] + patch_size[i], 1)
                for i in range(len(patch_size))
            ]
            + [slice(0, 1, 1)]
        )
        patch_slice.append(tuple(slice_list))

    # Return a list of slice
    return patch_slice
