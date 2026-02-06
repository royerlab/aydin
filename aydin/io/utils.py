"""Utility functions for image I/O operations.

This module provides helper functions for zarr file handling, output path
generation, channel splitting, and hyperstacking of image arrays.
"""

import os
from collections import Counter
from copy import deepcopy
from os.path import exists
from pathlib import Path
from typing import List, Tuple

import dask.array as da
import numpy
import zarr

from aydin.util.log.log import lprint


def is_zarr_storage(input_path):
    """Check if the given path points to a valid Zarr storage.

    Attempts to open the path as a Zarr array. If successful, the path
    is considered a Zarr storage.

    Parameters
    ----------
    input_path : str
        Path to check.

    Returns
    -------
    bool
        True if the path is a valid Zarr storage, False otherwise.
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
    """Read a Zarr file as a dask array.

    If the file is a ``zarr.Array``, reads it directly. If it is a
    ``zarr.Group``, reads it assuming dexp-zarr group format (stacking
    all non-MIP arrays along axis 0).

    Parameters
    ----------
    input_path : str
        Path to the Zarr file or directory.

    Returns
    -------
    array : dask.array.Array
        The image data as a dask array.
    """
    g = zarr.open(input_path, mode='r')
    if isinstance(g, zarr.Array):
        return da.from_zarr(input_path)
    else:
        arrays = [
            g[key][key] for key in g.group_keys() if "mip" not in g[key][key].name
        ]

        array = da.stack(arrays, axis=0)

        return array


def get_files_with_most_frequent_extension(path) -> List[str]:
    """Return files with the most common extension in a directory.

    Scans the given directory and returns all files whose extension
    matches the most frequently occurring extension.

    Parameters
    ----------
    path : str
        Path to the directory to scan.

    Returns
    -------
    files : List[str]
        List of filenames (not full paths) with the most common extension.
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
    """Generate a unique output file path for a given input image and operation.

    Appends the operation type to the filename and adds a numeric suffix
    if the output path already exists to avoid overwriting.

    Parameters
    ----------
    path : str
        Original input image file path.
    operation_type : str
        Type of operation. Supported values: ``'denoised'``, ``'hyperstacked'``.
    output_folder : str, optional
        If provided, places the output file in this folder instead of
        alongside the input.

    Returns
    -------
    output_path : str
        Unique output file path.
    counter : int or None
        Numeric suffix used to avoid collision, or None if no collision occurred.
    """
    if operation_type not in ["denoised", "hyperstacked"]:
        raise ValueError(
            f"invalud value for operation_type parameter: {operation_type}"
        )

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
    """Generate a path for saving denoising options as a JSON file.

    Parameters
    ----------
    path : str
        Original input image file path.
    passed_counter : int, optional
        Numeric suffix to use. If None, auto-increments to avoid overwriting.
    output_folder : str, optional
        If provided, places the options file in this folder.

    Returns
    -------
    options_path : str
        Path for the options JSON file.
    """
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
    """Generate a path for saving a trained denoiser model.

    Parameters
    ----------
    path : str
        Original input image file path.
    passed_counter : int, optional
        Numeric suffix to use. If None, auto-increments to avoid overwriting.
    output_folder : str, optional
        If provided, places the model in this folder.

    Returns
    -------
    model_path : str
        Path for the model directory.
    """
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
    """Split a multi-channel image into separate single-channel images.

    Splits along the channel axis ('C') and updates the metadata to
    reflect the reduced dimensionality.

    Parameters
    ----------
    image_array : numpy.typing.ArrayLike
        Multi-channel image array.
    metadata : FileMetadata
        Metadata for the image, must have a 'C' axis in ``metadata.axes``.

    Returns
    -------
    splitted_arrays : list of numpy.ndarray
        List of single-channel image arrays.
    metadatas : list of FileMetadata
        List of updated metadata objects, one per channel.
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

    metadatas = [deepcopy(metadata) for _ in splitted_arrays]

    return splitted_arrays, metadatas


def hyperstack_arrays(image_arrays, metadatas):
    """Stack multiple same-shape images into a single higher-dimensional array.

    Adds a new batch axis ('B') as the leading dimension and updates the
    metadata accordingly.

    Parameters
    ----------
    image_arrays : List[numpy.typing.ArrayLike]
        List of image arrays, all with the same shape.
    metadatas : List[FileMetadata]
        List of corresponding metadata objects.

    Returns
    -------
    image_array : numpy.ndarray
        Hyperstacked image array with a new leading batch axis.
    metadata : FileMetadata
        Updated metadata for the stacked image.

    Raises
    ------
    Exception
        If images have different shapes.
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
