"""Random patch extraction with entropy-based filtering.

Provides utilities to extract random patches from images and select
the most informative ones based on intensity histogram entropy.
"""

import numpy
from scipy.stats import entropy


def random_patches(
    image,
    patch_size: int,
    nb_patches_per_image: int,
    adoption_rate: float = 0.5,
):
    """Extract random patches from an image, filtered by entropy.

    Returns a list of slice objects for cropping patches from the image.
    Patches are sorted by their intensity histogram entropy and only those
    with higher entropy (more informative content) are retained.

    To work with any ``adoption_rate`` between 0 and 1, more patches are
    generated initially and then filtered down after entropy-based sorting.

    Parameters
    ----------
    image : numpy.ndarray
        Input image with axis order ``(B, C, Y, X)`` or ``(B, C, Z, Y, X)``.
    patch_size : int
        Spatial size of each patch along each dimension.
    nb_patches_per_image : int
        Desired number of patches to extract per batch element.
    adoption_rate : float
        Fraction of generated patches to keep after entropy sorting.
        Values between 0 and 1.

    Returns
    -------
    list of tuple of slice
        List of slicing tuples that can be used to index the input image
        to extract the selected patches.
    """
    list_of_slice_objects = []

    # Compute histogram range from the actual image data (guard constant images)
    img_min = float(image.min())
    img_max = float(image.max())
    if img_min == img_max:
        img_max = img_min + 1.0
    hist_range = (img_min, img_max)

    # Calculate total number of possible patches for a given image and patch_size
    possible_positions = numpy.asarray(image.shape[2:]) - patch_size + 1

    nb_possible_patches_per_image = numpy.prod(possible_positions)

    # Validate nb_patches_per_image, adoption_rate combination
    # is valid, if not generate all possible patches
    nb_patches_per_image = min(
        int(nb_patches_per_image / adoption_rate), nb_possible_patches_per_image
    )

    # print(nb_patches_per_image, nb_possible_patches_per_image, possible_positions)

    for b in range(image.shape[0]):  # b is a single element across batch dimension
        entropies = []
        slice_objects_for_current_b = []

        # Generate patches and entropy values
        while len(slice_objects_for_current_b) < nb_patches_per_image:
            indices_for_current_patch = [
                int(numpy.random.choice(s, 1)) for s in possible_positions
            ]
            slicing_for_current_patch = tuple(
                [
                    slice(b, b + 1, 1),
                    slice(0, 1, 1),
                    *[
                        slice(
                            x,
                            x + patch_size,
                            1,
                        )
                        for idx, x in enumerate(indices_for_current_patch)
                    ],
                ]
            )

            slice_objects_for_current_b.append(slicing_for_current_patch)

            current_patch = image[slicing_for_current_patch]

            # Calculate histogram and entropy
            hist, _ = numpy.histogram(
                current_patch, range=hist_range, bins=255, density=True
            )
            entropies.append(entropy(hist))

        # Sort patches
        sorted_indices = numpy.array(entropies).argsort()
        sorted_slice_objects = numpy.array(slice_objects_for_current_b)[sorted_indices]

        # Filter patches according to adoption_rate
        sorted_slice_objects = sorted_slice_objects[
            len(sorted_indices) - 1 - max(int(len(sorted_indices) * adoption_rate), 1) :
        ]

        # Have to convert each element to a tuple again so they
        # can be passed directly for slicing
        sorted_slice_objects = [tuple(elem) for elem in sorted_slice_objects]

        # Append the new patch slices to list_of_slice_objects
        list_of_slice_objects += sorted_slice_objects

    return list_of_slice_objects
