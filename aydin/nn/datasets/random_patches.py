import numpy
from scipy.stats import entropy


def random_patches(
    image,
    patch_size: int,
    nb_patches_per_image: int,
    adoption_rate: float = 0.5,
):
    """
    This functions returns list of slice objects that crops a part of the image
    which we call patch. Also sorts the patches, and makes sure only patches with
    higher entropy in the intensity histogram are selected.

    To be able to work with any adoption_rate between 0 and 1, we accordingly
    generate more patches per image during patch generation. After sorting, we
    are able to apply the adoption rate to the total number of patches we generated
    for each image.

    Parameters
    ----------
    image : numpy.ArrayLike
        This function assumes the axis order BXY(Z)C.
    patch_size : int
    nb_patches_per_image : int
    adoption_rate : float
    backend : str
        Option to choose axes convention for different backends, valid values: tensorflow, models

    Returns
    -------
    List of Tuples of Slicing Objects

    """
    list_of_slice_objects = []

    # Calculate total number of possible patches for a given image and patch_size
    possible_positions = numpy.asarray(image.shape[2:]) - patch_size + 1

    nb_possible_patches_per_image = numpy.prod(possible_positions)

    # Validate nb_patches_per_image, adoption_rate combination is valid, if not generate all possible patches
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
                current_patch, range=(0, 1), bins=255, density=True
            )
            entropies.append(entropy(hist))

        # Sort patches
        sorted_indices = numpy.array(entropies).argsort()
        sorted_slice_objects = numpy.array(slice_objects_for_current_b)[sorted_indices]

        # Filter patches according to adoption_rate
        sorted_slice_objects = sorted_slice_objects[
            len(sorted_indices) - 1 - max(int(len(sorted_indices) * adoption_rate), 1) :
        ]

        # Have to convert each element to a tuple again so they can be passed directly for slicing
        sorted_slice_objects = [tuple(elem) for elem in sorted_slice_objects]

        # Append the new patch slices to list_of_slice_objects
        list_of_slice_objects += sorted_slice_objects

    return list_of_slice_objects
