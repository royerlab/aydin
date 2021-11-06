def apply_slicing(data, slicing_string: str):
    """
    Function to apply slicing without evaluating argument passed by user

    Parameters
    ----------
    data : numpy.typing.ArrayLike
    slicing_string : str

    Returns
    -------
    sliced_data : numpy.typing.ArrayLike

    """
    if slicing_string == "":
        return data

    try:
        # Get slicing details for each dimension separately
        slices_for_dims = []
        slice_strings = slicing_string[1:-1].split(',')
        for _ in range(len(slice_strings)):  # Iterate over number of dimensions
            slice_string = slice_strings[_]

            # If it is not sliced on a dimension we build slice object to get entire dimension
            if slice_string == ":":
                slices_for_dims.append(slice(data.shape[_]))
            else:
                slice_values = slice_string.split(":")
                slices_for_dims.append(
                    slice(int(slice_values[0]), int(slice_values[1]))
                )

        # Create slice objects for each dimension and form the composite slice object from them
        slicer = [slice(None)] * len(slices_for_dims)
        for axis in range(len(slices_for_dims)):
            slicer[axis] = slices_for_dims[axis]

        # Apply resulting slice object to data array
        sliced_data = data[tuple(slicer)].copy()

    except Exception:
        raise ValueError("Passed slicing argument is not in valid...")

    # Return the sliced array back
    del data
    return sliced_data
