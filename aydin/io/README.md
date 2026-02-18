# Image I/O (`aydin/io/`)

This package handles multi-format image reading and writing for Aydin, with automatic format detection and metadata extraction.

## Architecture

```
aydin/io/
├── __init__.py     # Exports imread
├── io.py           # Core imread/imwrite functions, FileMetadata class
├── datasets.py     # Example image downloading (from Zenodo)
├── folders.py      # Platform-specific paths (cache, config, data)
└── utils.py        # I/O helper utilities
```

## Core API

### `imread(input_path)` → `(array, metadata)`

Reads an image file and returns a NumPy array with `FileMetadata`. Format is auto-detected from the file extension. Supports:

| Format | Extensions | Notes |
|--------|-----------|-------|
| TIFF | `.tif`, `.tiff` | Primary format; supports multi-dimensional stacks |
| CZI | `.czi` | Zeiss microscopy format |
| ND2 | `.nd2` | Nikon microscopy format |
| Zarr | `.zarr` | Chunked array storage |
| PNG | `.png` | Standard lossless image |
| JPEG | `.jpg`, `.jpeg` | Standard lossy image |
| NumPy | `.npy`, `.npz` | NumPy array format |

### `imwrite(array, output_path, metadata=None)`

Writes an array to disk. Format is determined by the output path extension.

### `FileMetadata`

Metadata container holding:
- `axes` — Dimension labels (e.g., `'TZYX'`)
- `shape` — Array shape
- `dtype` — Data type

The module also provides `is_batch()` and `is_channel()` helper functions for dimension classification.

### `mapped_tiff(output_path, shape, dtype)`

Creates a memory-mapped TIFF for out-of-core writing of large images.

## Additional Modules

- **`datasets.py`** — Downloads example images (e.g., noisy, ground-truth pairs) from Zenodo for testing and demos
- **`folders.py`** — Returns platform-specific directories for cache, config, and data storage

## For Contributors

To add support for a new image format:

1. Add a reader function in `io.py` (or a dedicated module)
2. Register the file extension in the format detection logic
3. Extract metadata (axes, shape, dtype) into `FileMetadata`
4. Add a corresponding writer if applicable

## Related Packages

- [`../gui/`](../gui/README.md) — GUI uses `imread`/`imwrite` for file loading and saving
- [`../cli/`](../cli/README.md) — CLI uses `imread`/`imwrite` for headless operations
- [`../analysis/`](../analysis/README.md) — Analysis functions operate on arrays loaded via I/O
