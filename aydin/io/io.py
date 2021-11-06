"""
Axis codes:
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
from czifile import czifile, CziFile
from nd2reader import ND2Reader
from numpy import array_equal
from tifffile import tifffile, TiffFile, memmap

from aydin.io.utils import is_zarr_storage, read_zarr_array
from aydin.util.log.log import lsection, lprint


def is_batch(code, shape, axes):
    """Method to check if given axis code belongs to a batch dimension.

    Parameters
    ----------
    code : str
    shape : tuple
    axes : str

    Returns
    -------
    bool

    """
    # special case:
    if len(shape) == 3 and 'X' in axes and 'Y' in axes and 'I' == code:
        return False

    return code not in 'XYZTQC'


def is_channel(code, length):
    """Method to check if given axis code belongs to channel dimension.

    Parameters
    ----------
    code :

    Returns
    -------
    bool

    """
    return code == "C" and not length > 8


class FileMetadata:
    """Metadata class used across aydin package.

    # TODO: make this docstring more detailed
    """

    def __init__(self):
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
        return f" is_folder={self.is_folder}, ext={self.extension}, axes={self.axes}, shape={self.shape}, batch_axes={self.batch_axes}, channel_axes={self.channel_axes}, dtype={self.dtype}, format={self.format} "

    def __eq__(self, other):
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
    """Image reading method.

    Method takes the image path as a string argument. Upon certain
    checks and decisions it returns the image array and its
    corresponding metadata.

    Parameters
    ----------
    input_path : str

    Returns
    -------
    tuple(numpy.typing.ArrayLike, FileMetadata)
        Returns tuple of (array, metadata).

    """

    with lsection(f"Reading image file at: {input_path}"):

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
                g = zarr.open(
                    input_path, mode='r'
                )  # TODO: fix this, it crashes with zarr arrays
                if isinstance(g, zarr.Array):
                    lprint(f"Reading file {input_path} as ZARR array")

                    if 'axes' in g.attrs:
                        metadata.axes = g.attrs['axes']
                else:
                    # Then we treat it as dexp-convention zarr group
                    lprint(f"Reading file {input_path} as ZARR group")
                    nb_arrays = 0
                    for key in g.group_keys():
                        nb_arrays += 1
                        if 'axes' in g[key][key].attrs:
                            metadata.axes = g.attrs['axes']

                metadata.format = 'zarr'
                array = read_zarr_array(input_path)
                metadata.shape = array.shape
                metadata.dtype = array.dtype

            elif is_tiff:
                lprint(f"Reading file {input_path} as TIFF file")
                with TiffFile(input_path) as tif:
                    if len(tif.series) >= 1:
                        serie = tif.series[0]
                        metadata.shape = serie.shape
                        metadata.dtype = serie.dtype
                        metadata.axes = serie.axes
                        metadata.other = tif.imagej_metadata
                    else:
                        lprint(f'There is no series in file: {input_path}')

                metadata.format = 'tiff'
                array = tifffile.imread(input_path)

            elif is_czi:
                lprint(f"Reading file {input_path} as CZI file")
                with CziFile(input_path) as czi:
                    metadata.format = 'czi'
                    metadata.axes = czi.axes
                    metadata.other = czi.metadata(raw=False)

                array = czifile.imread(input_path)
                metadata.shape = czi.shape
                metadata.dtype = czi.dtype

            elif is_png or is_jpg:
                lprint(f"Reading file {input_path} as PNG file")
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
                    lprint(
                        f"Warning: Can't interpret {'png' if is_png else 'jpg'} structure, might be incorrect!"
                    )
            elif is_npy:
                lprint(f"Reading file {input_path} as NPY file")

                array = numpy.load(input_path)
                metadata.format = 'npy'
                metadata.shape = array.shape
                metadata.dtype = array.dtype
                metadata.axes = ''.join(('Q',) * len(array.shape))

            elif is_npz:
                lprint(f"Reading file {input_path} as NPZ file")

                data = numpy.load(input_path)
                lprint(data.files)

                # this could contain several arrays, we read the one with the most voxels (good heuristic):
                # We read the largest array:
                biggest_size = 0
                for _file in data.files:
                    _array = data[_file]
                    size = numpy.size(_array)
                    lprint(
                        f"Reading array of name: {_file}, shape: {_array.shape}, and dtype: {_array.dtype}, size: {size}"
                    )

                    if biggest_size < size:
                        lprint("Bigger!")
                        file = _file
                        biggest_size = size
                        array = _array

                # makse sure the array is 'clean':
                array = numpy.asarray(array)
                lprint(
                    f"Selected array: name: {file}, shape: {array.shape}, and dtype: {array.dtype}"
                )
                metadata.format = 'npz'
                metadata.shape = array.shape
                metadata.dtype = array.dtype
                metadata.axes = ('TZYX' + ''.join(('Q',) * len(array.shape)))[
                    0 : array.ndim
                ]

            elif is_nd2:
                lprint(f"Reading file {input_path} as ND2 file")
                import pims

                n2image = ND2Reader(input_path)

                metadata.format = 'nd2'
                metadata.axes = ''.join(n2image.axes).upper()  # TODO: check order!

                n2image.bundle_axes = n2image.axes
                array = numpy.asarray(n2image[0], dtype=metadata.dtype)
                metadata.shape = array.shape
                metadata.dtype = array.dtype

            elif is_globlist:
                lprint(f"Reading file {input_path} as file list")
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
                    lprint(error)
                    lprint(traceback.format_exc())
                    lprint(
                        f"Tried to open file {input_path} with skimage io but failed to obtain image."
                    )
                    return None, None

        except Exception as error:
            lprint(error)
            lprint(traceback.format_exc())
            lprint(f"Could not read file {input_path} !")
            return None, None

        if metadata.axes:
            metadata.batch_axes = tuple(
                is_batch(axis, metadata.shape, metadata.axes) for axis in metadata.axes
            )

            metadata.channel_axes = tuple(
                is_channel(axis, s) for axis, s in zip(metadata.axes, metadata.shape)
            )

        lprint(f"Metadata: {metadata}")

        _sync_array_with_metadata(array, metadata)

        return array, metadata


def _sync_array_with_metadata(array, metadata):
    # We need to check if the metadata matches what we actually get, otherwise we need to update it.
    # This can happen for tiff files that are multi-part.
    if metadata is not None and array is not None:
        if metadata.shape != array.shape:
            metadata.shape = array.shape
        if metadata.dtype != array.dtype:
            metadata.dtype = array.dtype


def imwrite(array, output_path, metadata=None, overwrite=True):
    """Image writing method.

    Parameters
    ----------
    array : numpy.typing.ArrayLike
    output_path : str
    metadata : FileMetadata
    overwrite : bool

    """

    if not overwrite and exists(output_path):
        return

    if "png" in output_path and (
        len(array.shape) > 3
        or (len(array.shape) == 3 and array.shape[-1] not in [3, 4])
    ):
        lprint(
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
    """Internal method to write .tiff files with given array and metadata.

    Parameters
    ----------
    array : numpy.typing.ArrayLike
    output_path : str
    metadata : FileMetadata

    """
    # We get the ij metadata:
    ijmetadata = None if metadata is None else metadata.other

    # Normalise to {}:
    ijmetadata = {} if ijmetadata is None else ijmetadata

    tifffile.imwrite(output_path, array, metadata=ijmetadata)


@contextmanager
def mapped_tiff(output_path, shape, dtype):
    """Mapped tiff context manager.

    Parameters
    ----------
    output_path
    shape
    dtype

    Yields
    ------

    """
    array = memmap(output_path, shape=shape, dtype=dtype)
    try:
        yield array
        array.flush()
    finally:
        del array
        lprint(
            f"Flushing and writing all bytes to TIFF file {output_path}  (shape={shape}, dtype={dtype})"
        )
