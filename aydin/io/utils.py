import os
from collections import Counter
from copy import deepcopy
from os.path import exists
from pathlib import Path
from typing import List, Tuple
import numpy
import zarr
import dask.array as da

from aydin.util.log.log import lprint


def is_zarr_storage(input_path):
    """Method to check if given file is a zarr storage or not.

    Parameters
    ----------
    input_path : str

    Returns
    -------
    bool
        Result of whether the file in the given path is a zarr storage or not.

    """
    try:
        z = zarr.open(input_path)
        if len(z.shape) >= 0:
            lprint(f"This path is a ZARR storage: {input_path}")
        else:
            raise Exception
        # IF we reach this point, then we could open the file and therefore it is a Zarr file...
        return True
    except Exception:
        return False


def read_zarr_array(input_path):
    """Method that reads a zarr file. If the file is a zarr.Array, this will
    read as an zarr.Array. If the file is a zarr.Group, this method can only
    read it if it is in dexp-zarr group format.

    Parameters
    ----------
    input_path : str

    Returns
    -------
    numpy.typing.ArrayLike

    """
    g = zarr.open(input_path, mode='a')
    if isinstance(g, zarr.Array):
        return da.from_zarr(input_path)
    else:
        arrays = [
            g[key][key] for key in g.group_keys() if "mip" not in g[key][key].name
        ]

        array = da.stack(arrays, axis=0)

        return array


def get_files_with_most_frequent_extension(path) -> List[str]:
    """Method that looks into the given path and return the list of files with
    the most frequent file extension.

    Parameters
    ----------
    path : str

    Returns
    -------
    List[str]

    """

    files_in_folder = os.listdir(path)

    extensions = [Path(file).suffix[1:] for file in files_in_folder]

    counts = Counter(extensions)

    most_frequent_extension = sorted(counts, key=counts.__getitem__)[-1]

    files = [
        file for file in files_in_folder if file.endswith(f".{most_frequent_extension}")
    ]

    return files


def get_output_image_path(
    path: str, operation_type: str = "denoised", output_folder: str = None
) -> Tuple[str, int]:
    """Method to get correct output path for given input path and operation type.

    Parameters
    ----------
    path : str
    operation_type : str
        Currently supported values: 'denoised', 'deconvolved', 'hyperstacked'.
    output_folder : str

    Returns
    -------
    Tuple
        (Correct output path, counter).

    """

    if output_folder:
        path = os.path.join(output_folder, Path(path).name)

    image_formats = [
        ".zarr.zip",
        ".zarr",
        ".tiff",
        ".png",
        ".tif",
        ".TIF",
        ".czi",
        ".npy",
        ".nd2",
    ]

    for image_format in image_formats:
        if image_format in path:
            output_path = (
                f"{path.split(image_format)[0]}_{operation_type}{image_format}"
            )
            output_image_format = image_format
            break
    else:  # means no break in this context
        lprint("Image file format is not supported, will be writing result as tif")
        output_path = f"{path[:path.rfind('.')]}_{operation_type}.tif"
        output_image_format = ".tif"

    counter = 1
    response_counter = None
    while exists(output_path):
        response_counter = counter
        output_path = f"{output_path.split('_denoised')[0]}_{operation_type}{counter}{output_image_format}"
        counter += 1

    return output_path, response_counter


def get_options_json_path(
    path: str, passed_counter: int = None, output_folder: str = None
) -> str:
    if output_folder:
        path = os.path.join(output_folder, Path(path).name)

    options_path = f"{path[:path.rfind('.')]}_options.json"

    if passed_counter is None:
        counter = 1
        while exists(options_path):
            options_path = f"{options_path.split('_options')[0]}_options{counter}.json"
            counter += 1
    else:
        options_path = (
            f"{options_path.split('_options')[0]}_options{passed_counter}.json"
        )

    return options_path


def get_save_model_path(
    path: str, passed_counter: int = None, output_folder: str = None
) -> str:
    if output_folder:
        path = os.path.join(output_folder, Path(path).name)

    model_path = f"{path[:path.rfind('.')]}_model"

    if passed_counter is None:
        counter = 1
        while exists(model_path):
            model_path = f"{model_path.split('_model')[0]}_model{counter}"
            counter += 1
    else:
        model_path = f"{model_path.split('_model')[0]}_model{passed_counter}"

    return model_path


def split_image_channels(image_array, metadata):
    """Method that takes a multichannel image and its metadata and splits
    into single channel images.

    Parameters
    ----------
    image_array : numpy.typing.ArrayLike
    metadata : FileMetadata

    Returns
    -------
    tuple(List[numpy.typing.ArrayLike], List[FileMetadata])
        Tuple of splitted_arrays and metadatas.

    """
    channel_axis = metadata.axes.find("C")

    if channel_axis == -1:
        lprint("Array has no channel axis detected")
        return

    # Handle image splitting
    splitted_arrays = numpy.split(
        image_array, metadata.shape[channel_axis], axis=channel_axis
    )
    splitted_arrays = [numpy.squeeze(array) for array in splitted_arrays]

    # Handle metadata changes
    metadata.batch_axes = tuple(
        x for ind, x in enumerate(metadata.batch_axes) if ind != channel_axis
    )
    metadata.channel_axes = tuple(
        x for ind, x in enumerate(metadata.channel_axes) if ind != channel_axis
    )
    metadata.axes = metadata.axes.replace("C", "")
    metadata.shape = tuple(
        x for idx, x in enumerate(metadata.shape) if idx != channel_axis
    )
    metadata.splitted = True

    metadatas = [metadata] * len(splitted_arrays)

    return splitted_arrays, metadatas


def hyperstack_arrays(image_arrays, metadatas):
    """Method that takes a list of arrays of same shape and their corresponding
    metadatas, then hyperstacks those into a single image.

    Parameters
    ----------
    image_arrays : List[numpy.typing.ArrayLike]
    metadatas : List[FileMetadata]

    Returns
    -------
    tuple(numpy.typing.ArrayLike, FileMetadata)
        Tuple of hyperstacked image array and its metadata.

    """
    if len(image_arrays) < 2:
        lprint("Need at least two images to hyperstack.")
        return image_arrays, metadatas

    shape_of_first_image = ()

    for idx, metadata in enumerate(metadatas):
        if idx == 0:
            shape_of_first_image = metadata.shape
        elif shape_of_first_image != metadata.shape:
            raise Exception(
                "Images are not same shape, hence cannot hyperstack images."
            )

    metadata = deepcopy(metadatas[-1])

    metadata.axes = "B" + metadata.axes
    metadata.batch_axes = (True,) + metadata.batch_axes
    metadata.channel_axes = (False,) + metadata.channel_axes
    metadata.shape = (len(image_arrays),) + metadata.shape
    image_array = numpy.stack(image_arrays)

    return image_array, metadata
