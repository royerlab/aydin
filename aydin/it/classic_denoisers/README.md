# Classical Denoisers (`aydin/it/classic_denoisers/`)

This subpackage provides 14 classical image denoising algorithms, each with automatic parameter calibration via J-invariance (Noise2Self).

## Algorithms

### Frequency-Domain

| Module | Algorithm | Description |
|--------|-----------|-------------|
| `butterworth.py` | Butterworth low-pass | Smooth frequency cutoff filter with configurable order |
| `gaussian.py` | Gaussian low-pass | Standard Gaussian smoothing in frequency domain |
| `spectral.py` | Spectral thresholding | DCT/DST/FFT-based patch coefficient thresholding |
| `wavelet.py` | Wavelet thresholding | BayesShrink and VisuShrink wavelet denoising |

### Patch-Based

| Module | Algorithm | Description |
|--------|-----------|-------------|
| `nlm.py` | Non-Local Means | Weighted average of similar patches across the image |
| `bmnd.py` | Block-Matching nD | Generalized BM3D for arbitrary dimensions |

### Optimization-Based

| Module | Algorithm | Description |
|--------|-----------|-------------|
| `tv.py` | Total Variation | Bregman and Chambolle TV regularization |
| `harmonic.py` | Harmonic prior | Non-linear smoothing via harmonic energy minimization |

### Edge-Preserving

| Module | Algorithm | Description |
|--------|-----------|-------------|
| `bilateral.py` | Bilateral filter | Spatially-weighted filter that preserves edges |
| `lipschitz.py` | Lipschitz continuity | Constrains gradient magnitude; effective against impulse noise |

### Hybrid

| Module | Algorithm | Description |
|--------|-----------|-------------|
| `gm.py` | Gaussian-Median mix | Blends Gaussian smoothing with median filtering |

### Dictionary-Based

| Module | Algorithm | Description |
|--------|-----------|-------------|
| `pca.py` | PCA patch denoising | Denoises via PCA projection of image patches |
| `dictionary_fixed.py` | Fixed dictionary | Sparse coding over fixed DCT/DST dictionary |
| `dictionary_learned.py` | Learned dictionary | Sparse coding over dictionary learned from the image |

### Support

- `_defaults.py` — Default parameters shared across denoisers

## Key Patterns

### Dual-Function Interface

Every denoiser exposes two public functions:

```python
# 1. Auto-calibrate parameters using J-invariance, then denoise
best_params, denoised, quality = calibrate_denoise_X(image, ...)

# 2. Denoise with explicit parameters (no calibration)
denoised = denoise_X(image, ...)
```

The `calibrate_denoise_*` functions search over parameter grids, evaluating each configuration via the J-invariance self-supervised loss. They return the best parameters found, the denoised result, and a quality score.

### Integration with `ImageDenoiserClassic`

`ImageDenoiserClassic` (in `../classic.py`) discovers all `calibrate_denoise_*` functions in this subpackage at runtime and exposes them as algorithm variants. Each channel is calibrated independently, using the highest-variance batch for optimization.

## Important: HTML Docstrings

The `denoise_*` function docstrings contain HTML tags rendered by the Aydin GUI:
- **`<a href="...">`** — Wikipedia links for algorithm names (clickable in GUI)
- **`<notgui>`** — Separates GUI-visible description from API-only Parameters section
- **`\n\n`** — Paragraph breaks converted to `<br><br>` at runtime

**Do not remove or convert these tags.** See [CLAUDE.md](../../../CLAUDE.md) for full docstring conventions.

## For Contributors

To add a new classical denoiser:

1. Create `my_denoiser.py` in this directory
2. Implement `denoise_my_denoiser(image, **params)` and `calibrate_denoise_my_denoiser(image, ...)`
3. Follow the dual-function pattern — `calibrate_denoise_*` should call `denoise_*` internally
4. Add `<a href>` links and `<notgui>` tags in docstrings following existing examples
5. The algorithm will be auto-discovered by `ImageDenoiserClassic`

## Related Packages

- [`../`](../README.md) — Parent Image Translator framework
- [`../../restoration/denoise/`](../../restoration/denoise/README.md) — High-level `Classic` wrapper that uses these denoisers
