"""Image reading and writing for multiple file formats.

This module provides ``imread`` and ``imwrite`` functions that support TIFF,
CZI, PNG, JPEG, NPY, NPZ, ND2, Zarr, and file glob patterns. Metadata
(axes, shape, dtype, batch/channel classification) is extracted automatically.

Axis codes used in metadata:
    'X' width, 'Y' height, 'S' sample, 'I' image series|page|plane,
    'Z' depth, 'C' color|em-wavelength|channel, 'E' ex-wavelength|lambda,
    'T' time, 'R' region|tile, 'A' angle, 'P' phase, 'H' lifetime,
    'L' exposure, 'V' event, 'Q' unknown, '_' missing
"""

import os
import pathlib
import traceback
from contextlib import contextmanager
from os.path import exists
from pathlib import Path

import numpy
import skimage
import zarr
from czifile import CziFile, czifile
from nd2reader import ND2Reader
from numpy import array_equal
from tifffile import TiffFile, memmap, tifffile

from aydin.io.utils import is_zarr_storage, read_zarr_array
from aydin.util.log.log import aprint, asection


def is_batch(code, shape, axes):
    """Check if a given axis code belongs to a batch dimension.

    An axis is considered a batch dimension if its code is not one of the
    spatial/temporal codes (X, Y, Z, T, Q, C), with a special case for
    the 'I' code in 3D XY images.

    Parameters
    ----------
    code : str
        Single-character axis code (e.g., 'X', 'Y', 'T', 'I').
    shape : tuple of int
        Shape of the image array.
    axes : str
        String of axis codes for all dimensions (e.g., 'TZCYX').

    Returns
    -------
    bool
        True if the axis is a batch dimension.
    """
    # special case:
    if len(shape) == 3 and 'X' in axes and 'Y' in axes and 'I' == code:
        return False

    return code not in 'XYZTQC'


def is_channel(code, length):
    """Check if a given axis code belongs to a channel dimension.

    An axis is considered a channel dimension if its code is 'C' and
    the dimension length is 8 or fewer.

    Parameters
    ----------
    code : str
        Single-character axis code.
    length : int
        Length of the axis (number of entries along this dimension).

    Returns
    -------
    bool
        True if the axis is a channel dimension.
    """
    return code == "C" and not length > 8


class FileMetadata:
    """Metadata container for image files used across the Aydin package.

    Stores information about the image file including format, shape, dtype,
    axis codes, and the classification of each axis as batch, channel, or
    spatial/temporal.

    Attributes
    ----------
    is_folder : bool or None
        Whether the source path is a directory.
    extension : str or None
        File extension (without leading dot), lowercased.
    axes : str or None
        String of axis codes (e.g., 'TZCYX').
    shape : tuple of int or None
        Shape of the image array.
    dtype : numpy.dtype or None
        Data type of the image.
    format : str or None
        Image format identifier (e.g., 'tiff', 'zarr', 'czi').
    batch_axes : tuple of bool or None
        Boolean tuple indicating which axes are batch dimensions.
    channel_axes : tuple of bool or None
        Boolean tuple indicating which axes are channel dimensions.
    other : dict or None
        Additional format-specific metadata (e.g., ImageJ metadata).
    splitted : bool
        Whether the image has been split along the channel axis.
    """

    def __init__(self):
        """Initialize a FileMetadata instance with all attributes set to None.

        All attributes default to None (or False for ``splitted``), and are
        populated after reading an image file via :func:`imread`.
        """
        self.is_folder = None
        self.extension = None
        self.axes = None
        self.shape = None
        self.dtype = None
        self.format = None
        self.batch_axes = None
        self.channel_axes = None
        self.other = None
        self.splitted = False

    def __str__(self) -> str:
        """Return a human-readable string representation of the metadata.

        Returns
        -------
        str
            Formatted string showing all metadata fields.
        """
        return f" is_folder={self.is_folder}, ext={self.extension}, axes={self.axes}, shape={self.shape}, batch_axes={self.batch_axes}, channel_axes={self.channel_axes}, dtype={self.dtype}, format={self.format} "

    def __eq__(self, other):
        """Check equality between two FileMetadata instances.

        Compares all metadata fields except ``splitted``.

        Parameters
        ----------
        other : FileMetadata
            The other metadata object to compare against.

        Returns
        -------
        bool or NotImplemented
            True if all compared fields are equal, False otherwise.
            Returns ``NotImplemented`` if ``other`` is not a
            :class:`FileMetadata` instance.
        """
        if not isinstance(other, FileMetadata):
            return NotImplemented  # don't attempt to compare against unrelated types

        return (
            self.is_folder == other.is_folder
            and self.extension == other.extension
            and self.axes == other.axes
            and self.shape == other.shape
            and self.dtype == other.dtype
            and self.format == other.format
            and self.batch_axes == other.batch_axes
            and self.channel_axes == other.channel_axes
            and self.other == other.other
        )


def imread(input_path):
    """Read an image file and return its array data and metadata.

    Supports TIFF, CZI, PNG, JPEG, NPY, NPZ, ND2, Zarr, and file
    glob patterns. Automatically detects the file format from the
    extension and extracts axis, shape, and dtype metadata.

    Parameters
    ----------
    input_path : str
        Path to the image file or directory.

    Returns
    -------
    array : numpy.ndarray or None
        The image data as a NumPy array. None if reading fails.
    metadata : FileMetadata or None
        Metadata about the image file. None if reading fails.
    """

    with asection(f"Reading image file at: {input_path}"):

        metadata = FileMetadata()

        metadata.is_folder = os.path.isdir(input_path)
        metadata.extension = ((Path(input_path).suffix)[1:]).lower()

        is_tiff = 'tif' in metadata.extension or 'tiff' in metadata.extension
        is_czi = 'czi' in metadata.extension
        is_png = 'png' in metadata.extension
        is_jpg = 'jpg' in metadata.extension or 'jpeg' in metadata.extension
        is_zarr = 'zarr' in metadata.extension or is_zarr_storage(input_path)
        is_npy = 'npy' in metadata.extension
        is_npz = 'npz' in metadata.extension
        is_nd2 = 'nd2' in metadata.extension
        is_globlist = '*' in input_path

        try:
            if is_zarr:
                g = zarr.open(input_path, mode='r')
                if isinstance(g, zarr.Array):
                    aprint(f"Reading file {input_path} as ZARR array")

                    if 'axes' in g.attrs:
                        metadata.axes = g.attrs['axes']
                else:
                    # Then we treat it as dexp-convention zarr group
                    aprint(f"Reading file {input_path} as ZARR group")
                    nb_arrays = 0
                    for key in g.group_keys():
                        nb_arrays += 1
                        if 'axes' in g[key][key].attrs:
                            metadata.axes = g[key][key].attrs['axes']

                metadata.format = 'zarr'
                array = read_zarr_array(input_path)
                metadata.shape = array.shape
                metadata.dtype = array.dtype

            elif is_tiff:
                aprint(f"Reading file {input_path} as TIFF file")
                with TiffFile(input_path) as tif:
                    if len(tif.series) >= 1:
                        serie = tif.series[0]
                        metadata.shape = serie.shape
                        metadata.dtype = serie.dtype
                        metadata.axes = serie.axes
                        metadata.other = tif.imagej_metadata
                    else:
                        aprint(f'There is no series in file: {input_path}')

                metadata.format = 'tiff'
                array = tifffile.imread(input_path)

            elif is_czi:
                aprint(f"Reading file {input_path} as CZI file")
                with CziFile(input_path) as czi:
                    metadata.format = 'czi'
                    metadata.axes = czi.axes
                    metadata.other = czi.metadata(raw=False)
                    metadata.shape = czi.shape
                    metadata.dtype = czi.dtype

                array = czifile.imread(input_path)

            elif is_png or is_jpg:
                aprint(f"Reading file {input_path} as PNG file")
                array = skimage.io.imread(input_path)

                # We check if this is a gray level image:
                if len(array.shape) == 3:
                    if array_equal(array[..., 0], array[..., 1]) and array_equal(
                        array[..., 0], array[..., 2]
                    ):
                        # We keep the first channel only:
                        array = array[..., 0]

                metadata.format = 'png' if is_png else 'jpg'
                metadata.shape = array.shape
                metadata.dtype = array.dtype

                if len(array.shape) == 2:
                    metadata.axes = "YX"
                elif len(array.shape) == 3 and array.shape[-1] in (3, 4):
                    metadata.axes = "YXC"
                elif len(array.shape) == 3:
                    metadata.axes = "ZYX"
                elif len(array.shape) == 4 and array.shape[-1] in (3, 4):
                    metadata.axes = "ZYXC"
                else:
                    metadata.axes = "ZYXC"
                    aprint(
                        f"Warning: Can't interpret {'png' if is_png else 'jpg'} structure, might be incorrect!"
                    )
            elif is_npy:
                aprint(f"Reading file {input_path} as NPY file")

                array = numpy.load(input_path)
                metadata.format = 'npy'
                metadata.shape = array.shape
                metadata.dtype = array.dtype
                metadata.axes = ''.join(('Q',) * len(array.shape))

            elif is_npz:
                aprint(f"Reading file {input_path} as NPZ file")

                data = numpy.load(input_path)
                aprint(data.files)

                # this could contain several arrays, we read the one with the most voxels (good heuristic):
                # We read the largest array:
                biggest_size = 0
                file = None
                for _file in data.files:
                    _array = data[_file]
                    size = numpy.size(_array)
                    aprint(
                        f"Reading array of name: {_file}, shape: {_array.shape}, and dtype: {_array.dtype}, size: {size}"
                    )

                    if biggest_size < size:
                        aprint("Bigger!")
                        file = _file
                        biggest_size = size
                        array = _array

                # makse sure the array is 'clean':
                array = numpy.asarray(array)
                aprint(
                    f"Selected array: name: {file}, shape: {array.shape}, and dtype: {array.dtype}"
                )
                metadata.format = 'npz'
                metadata.shape = array.shape
                metadata.dtype = array.dtype
                metadata.axes = ('TZYX' + ''.join(('Q',) * len(array.shape)))[
                    0 : array.ndim
                ]

            elif is_nd2:
                aprint(f"Reading file {input_path} as ND2 file")
                import pims

                n2image = ND2Reader(input_path)

                metadata.format = 'nd2'
                metadata.axes = ''.join(n2image.axes).upper()  # TODO: check order!

                n2image.bundle_axes = n2image.axes
                array = numpy.asarray(n2image[0], dtype=metadata.dtype)
                metadata.shape = array.shape
                metadata.dtype = array.dtype

            elif is_globlist:
                aprint(f"Reading file {input_path} as file list")
                import pims

                array = pims.ImageSequence(input_path)
                metadata.format = 'globlist'
                metadata.shape = array.shape
                metadata.dtype = array.dtype
                metadata.axes = ('TZYX' + ''.join(('Q',) * len(array.shape)))[
                    0 : array.ndim
                ]

            # elif is_folder:
            #     from aydin.io import io
            #
            #     files = get_files_with_most_frequent_extension(input_path)
            #     files.sort()
            #
            #     imread = dask.delayed(io.imread, pure=True)  # Lazy version of imread
            #
            #     lazy_images = [
            #         imread(join(input_path, filename)) for filename in files
            #     ]  # Lazily evaluate imread on each path
            #
            #     file_metadata = analyse(join(input_path, files[0]))
            #
            #     arrays = [
            #         dask.array.from_delayed(
            #             lazy_image,  # Construct a small Dask array
            #             dtype=file_metadata.dtype,  # for every lazy value
            #             shape=file_metadata.shape,
            #         )
            #         for lazy_image in lazy_images
            #     ]
            #
            #     array = dask.array.stack(arrays, axis=0)
            #
            #     metadata.format = 'folder-' + file_metadata.format
            #     metadata.shape = array.shape
            #     metadata.dtype = array.dtype
            #     metadata.axes = 'Q' + file_metadata.axes
            #
            #     metadata.array = array
            #
            #     pass

            else:
                try:
                    array = skimage.io.imread(input_path)
                    metadata.format = pathlib.Path(input_path).suffix
                    metadata.shape = array.shape
                    metadata.dtype = array.dtype
                    metadata.axes = ('TZYX' + ''.join(('Q',) * len(array.shape)))[
                        0 : array.ndim
                    ]
                except Exception as error:
                    aprint(error)
                    aprint(traceback.format_exc())
                    aprint(
                        f"Tried to open file {input_path} with skimage io but failed to obtain image."
                    )
                    return None, None

        except Exception as error:
            aprint(error)
            aprint(traceback.format_exc())
            aprint(f"Could not read file {input_path} !")
            return None, None

        if metadata.axes:
            metadata.batch_axes = tuple(
                is_batch(axis, metadata.shape, metadata.axes) for axis in metadata.axes
            )

            metadata.channel_axes = tuple(
                is_channel(axis, s) for axis, s in zip(metadata.axes, metadata.shape)
            )

        aprint(f"Metadata: {metadata}")

        _sync_array_with_metadata(array, metadata)

        return array, metadata


def _sync_array_with_metadata(array, metadata):
    """Synchronize metadata with the actual array shape and dtype.

    Updates metadata if the actual array shape or dtype differs from
    what was reported in the file headers (e.g., for multi-part TIFF files).

    Parameters
    ----------
    array : numpy.ndarray or None
        The loaded image array.
    metadata : FileMetadata or None
        Metadata to update in place.
    """
    # We need to check if the metadata matches what we actually get, otherwise we need to update it.
    # This can happen for tiff files that are multi-part.
    if metadata is not None and array is not None:
        if metadata.shape != array.shape:
            metadata.shape = array.shape
        if metadata.dtype != array.dtype:
            metadata.dtype = array.dtype


def imwrite(array, output_path, metadata=None, overwrite=True):
    """Write an image array to a file.

    Automatically handles format selection based on the output path extension.
    Falls back to TIFF format if the requested format fails or is unsupported.

    Parameters
    ----------
    array : numpy.typing.ArrayLike
        Image data to write.
    output_path : str
        Destination file path. Format is inferred from the extension.
    metadata : FileMetadata, optional
        Metadata to include in the output file (used for TIFF files).
    overwrite : bool
        If False, skips writing when the output file already exists.
    """

    if not overwrite and exists(output_path):
        return

    if "png" in output_path and (
        len(array.shape) > 3
        or (len(array.shape) == 3 and array.shape[-1] not in [3, 4])
    ):
        aprint(
            "png images with more than 2 dimensions are not supported, will be writing the result as a tif"
        )
        output_path = f"{output_path[:output_path.rfind('.')]}.tif"

    if output_path[-3:] == "tif":
        _write_tiff(output_path, array, metadata)
    else:
        try:
            skimage.io.imsave(output_path, array)
        except Exception:
            # if skimage.io.imsave fails for any reason we fallback to .tif format
            output_path = f"{output_path[:output_path.rfind('.')]}.tif"
            _write_tiff(output_path, array, metadata)


def _write_tiff(output_path, array, metadata):
    """Write an array to a TIFF file with optional ImageJ metadata.

    Parameters
    ----------
    output_path : str
        Destination file path.
    array : numpy.typing.ArrayLike
        Image data to write.
    metadata : FileMetadata or None
        Metadata containing ImageJ-compatible metadata in its ``other`` attribute.
    """
    # We get the ij metadata:
    ijmetadata = None if metadata is None else metadata.other

    # Normalise to {}:
    ijmetadata = {} if ijmetadata is None else ijmetadata

    tifffile.imwrite(output_path, array, metadata=ijmetadata)


@contextmanager
def mapped_tiff(output_path, shape, dtype):
    """Context manager for memory-mapped TIFF file writing.

    Creates a memory-mapped TIFF file that can be written to incrementally.
    The file is flushed and finalized when the context manager exits.

    Parameters
    ----------
    output_path : str
        Destination file path for the TIFF file.
    shape : tuple of int
        Shape of the output image array.
    dtype : numpy.dtype
        Data type of the output image.

    Yields
    ------
    array : numpy.ndarray
        Memory-mapped array that can be written to.
    """
    array = memmap(output_path, shape=shape, dtype=dtype)
    try:
        yield array
        array.flush()
    finally:
        del array
        aprint(
            f"Flushing and writing all bytes to TIFF file {output_path}  (shape={shape}, dtype={dtype})"
        )
