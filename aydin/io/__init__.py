"""Image I/O module supporting multiple file formats.

This package provides functions for reading and writing n-dimensional images
in various formats including TIFF, CZI, PNG, JPEG, NPY, NPZ, ND2, and Zarr.
The primary entry point is the :func:`imread` function, which automatically
detects the file format and extracts metadata (axes, shape, dtype).

Submodules
----------
io
    Core ``imread`` and ``imwrite`` functions with format auto-detection.
datasets
    Example images for testing and demonstration, downloaded from Google Drive.
folders
    Platform-specific directory paths (home, temp, cache).
utils
    Helper functions for Zarr handling, output path generation, channel
    splitting, and hyperstacking.
"""

# flake8: noqa
from aydin.io.io import imread
