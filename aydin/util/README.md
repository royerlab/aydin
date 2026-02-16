# Utilities (`aydin/util/`)

This package contains 22 utility subpackages providing image processing, performance optimization, and infrastructure support used throughout Aydin.

## Subpackages by Category

### Logging and Strings

| Subpackage | Purpose |
|------------|---------|
| `log/` | Logging system — `aprint()` and `asection()` (from the arbol library) replace standard print |
| `string/` | String utilities — includes `strip_notgui()` in `break_text.py` for removing `<notgui>` tags from docstrings |

### Array and Image Processing

| Subpackage | Purpose |
|------------|---------|
| `array/` | NumPy array manipulation utilities |
| `crop/` | Image cropping operations |
| `denoise_nd/` | N-dimensional denoising helpers |
| `edge_filter/` | Edge detection filters |
| `patch_size/` | Optimal patch size estimation |
| `patch_transform/` | Patch extraction and reconstruction |
| `psf/` | Point spread function utilities |

### Performance-Optimized Operations

| Subpackage | Purpose |
|------------|---------|
| `fast_correlation/` | Optimized correlation computation |
| `fast_shift/` | Optimized image shift operations |
| `fast_uniform_filter/` | Optimized uniform (box) filtering |
| `numpy_memoize/` | NumPy function memoization/caching |
| `offcore/` | Off-core (memory-mapped) array computation for large images |

### Algorithm Support

| Subpackage | Purpose |
|------------|---------|
| `blindspot/` | Blind-spot utilities for self-supervised denoising |
| `bm3d/` | BM3D denoiser wrapper |
| `dictionary/` | Dictionary learning for sparse coding denoisers |
| `j_invariance/` | J-invariance (Noise2Self) loss computation |
| `nsi/` | Noise scale independence utilities |
| `optimizer/` | Optimization utilities (parameter search) |

### Infrastructure

| Subpackage | Purpose |
|------------|---------|
| `torch/` | PyTorch utilities — `device.py` for GPU/CPU device selection |
| `misc/` | Miscellaneous helpers |

## Key Modules

### `log/log.py`

The standard logging API for Aydin:

```python
from aydin.util.log.log import aprint, asection

aprint("Status message")
with asection("Processing step"):
    aprint("Detail")
```

All modules should use `aprint`/`asection` instead of `print()`.

### `string/break_text.py`

Contains `strip_notgui(text)` — strips everything after `<notgui>` from docstrings. Used by the GUI to separate user-visible content from API-only documentation.

### `torch/device.py`

Handles PyTorch device selection (CPU vs CUDA GPU), used by CNN-based denoising.

## For Contributors

To add a new utility subpackage:

1. Create a new directory under `util/` with an `__init__.py`
2. Add tests in a `tests/` subdirectory
3. Use `aprint`/`asection` from `log/log.py` for any logging

## Related Packages

- [`../it/`](../it/README.md) — Core framework uses J-invariance, blind-spot, and fast-filter utilities
- [`../features/`](../features/README.md) — Feature groups use fast_correlation, fast_uniform_filter, and offcore
- [`../nn/`](../nn/README.md) — Neural network package uses torch device utilities
- [`../gui/`](../gui/README.md) — GUI uses string utilities for docstring processing
