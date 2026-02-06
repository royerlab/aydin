"""Off-core array allocation for large datasets.

Provides memory-mapped and zarr-backed array allocation that
automatically falls back to disk-based storage when physical
memory is insufficient.
"""

import tempfile
from typing import Generator, Tuple, Union

import numpy
import psutil

from aydin.util.log.log import lprint, lsection


class OffCore:
    """Configuration for off-core array storage.

    Attributes
    ----------
    memmap_directory : str or None
        Directory for memory-mapped temporary files. If None, the
        system default temporary directory is used.
    """

    memmap_directory = None


def offcore_array(
    shape: Union[Tuple[int, ...], Generator[int, None, None]],
    dtype: numpy.dtype,
    force_memmap: bool = False,
    zarr_allowed: bool = False,
    no_memmap_limit: bool = True,
    max_memory_usage_ratio: float = 0.9,
):
    """Allocate an array, using off-core storage when memory is limited.

    Checks available physical and swap memory, and falls back to
    memory-mapped files or zarr-backed arrays when the requested
    array would exceed available memory.

    Parameters
    ----------
    shape : tuple of int
        Shape of the array to allocate.
    dtype : numpy.dtype
        Data type of the array.
    force_memmap : bool
        If True, always use memory-mapped storage regardless of
        available memory.
    zarr_allowed : bool
        If True, allow zarr-backed storage as a fallback when
        memory-mapping is not available.
    no_memmap_limit : bool
        If True, use memory-mapping even for very large arrays
        that exceed total system memory.
    max_memory_usage_ratio : float
        Maximum fraction of total memory (physical + swap) to use
        before switching to off-core storage.

    Returns
    -------
    numpy.ndarray or numpy.memmap or zarr.Array
        Allocated array using the most appropriate storage backend.
    """

    with lsection(f"Array of shape: {shape} and dtype: {dtype} requested"):
        size_in_bytes = numpy.prod(shape) * numpy.dtype(dtype).itemsize
        lprint(f'Array requested will be {(size_in_bytes / 1E6)} MB.')

        total_physical_memory_in_bytes = psutil.virtual_memory().total
        total_swap_memory_in_bytes = psutil.swap_memory().total

        total_mem_in_bytes = total_physical_memory_in_bytes + total_swap_memory_in_bytes
        lprint(
            f'There is {int(psutil.virtual_memory().total / 1E6)} MB of physical memory'
        )
        lprint(f'There is {int(psutil.swap_memory().total / 1E6)} MB of swap memory')
        lprint(f'There is {int(total_mem_in_bytes / 1E6)} MB of total memory')

        is_enough_total_memory = (
            size_in_bytes < max_memory_usage_ratio * total_mem_in_bytes
        )

        if not force_memmap and is_enough_total_memory:
            lprint(
                'There is enough physical+swap memory -- we do not need to use a mem mapped array or zarr-backed array.'
            )
            array = numpy.zeros(shape, dtype=dtype)

        elif no_memmap_limit:
            lprint(
                'There is not enough physical+swap memory -- we will use a mem mapped array.'
            )
            temp_file = tempfile.NamedTemporaryFile(dir=OffCore.memmap_directory)
            lprint(
                f'The temporary memory mapped file is at: {temp_file.name} (but you might not be able to see it!)'
            )
            array = numpy.memmap(temp_file, dtype=dtype, mode='w+', shape=shape)

        elif zarr_allowed:
            lprint(
                'There is not enough physical+swap memory -- we will use a zarr-backed array.'
            )
            import zarr

            array = zarr.create(
                shape=shape, dtype=dtype, store=zarr.TempStore("output.zarr")
            )
            # from numcodecs import Blosc
            # compressor = Blosc(cname = 'zstd', clevel = 3, shuffle = Blosc.BITSHUFFLE)
            # array = zarr.zeros((102_0, 200, 210), chunks = (100, 200, 210), compressor = compressor

        return array
