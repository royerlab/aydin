"""Off-core array allocation for large datasets.

This subpackage provides functions for allocating arrays that
automatically fall back to memory-mapped files or zarr-backed storage
when physical memory is insufficient for the requested array size.
"""
