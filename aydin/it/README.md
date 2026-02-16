# Image Translator Framework (`aydin/it/`)

The Image Translator (IT) framework is the core abstraction layer for all denoising in Aydin. It handles multi-dimensional image arrays with batch/channel/spatial dimensions, tiled processing, transform pipelines, normalization, and model serialization.

## Architecture

### Base Class

`ImageTranslatorBase` (`base.py`) is the abstract base class that all translators implement. Key responsibilities:

- **Dimension handling** — Interprets batch, channel, and spatial axes from n-dimensional arrays
- **Tiled processing** — Splits large images into tiles with configurable margins to fit in memory
- **Transform pipeline** — Applies an ordered list of `ImageTransformBase` transforms before/after denoising
- **Normalization** — Per-channel value normalization via normaliser classes
- **Training/inference** — `train()` and `translate()` public API with internal `_train()` / `_translate()` hooks
- **Serialization** — `save()` / `load()` for model persistence via JSON pickle

### Implementations

| Module | Class | Approach |
|--------|-------|----------|
| `classic.py` | `ImageDenoiserClassic` | Wraps classical denoising algorithms with J-invariance auto-calibration |
| `fgr.py` | `ImageTranslatorFGR` | Feature Generation & Regression — combines multi-scale features with gradient boosting (recommended) |
| `cnn_torch.py` | `ImageTranslatorCNNTorch` | PyTorch CNN with pluggable architectures (UNet, JiNet, DnCNN) |

Additional module:
- `timelapse_denoiser.py` — Specializes time-lapse denoising by exploiting temporal redundancy

### Subpackages

| Subpackage | Purpose |
|------------|---------|
| [`classic_denoisers/`](classic_denoisers/README.md) | 14 classical denoising algorithms with auto-calibration |
| [`transforms/`](transforms/README.md) | 11 image preprocessing/postprocessing transforms |
| `normalisers/` | Value normalization strategies (identity, min-max, percentile, shape) |
| `balancing/` | Data histogram balancing for regression training |
| `exceptions/` | Custom exception classes |

## Key Patterns

### Subclassing `ImageTranslatorBase`

To add a new translator, subclass `ImageTranslatorBase` and implement:

```python
class MyTranslator(ImageTranslatorBase):
    def _train(self, input_image, target_image, ...):
        """Train on a single image pair."""
        ...

    def _translate(self, input_image, ...):
        """Denoise a single image."""
        ...

    def _load_internals(self, path):
        """Restore model-specific state from disk."""
        ...
```

### J-Invariance

All self-supervised approaches use J-invariance (Noise2Self) — the denoised value for each pixel is computed without using that pixel's own noisy value. This is handled via blind spots in `ImageTranslatorBase`.

### Tiling

Large images are automatically tiled when they exceed `max_voxels_per_tile` (default 768^3). Tile margins ensure border artifacts are minimized. The tiling logic lives in the base class.

## For Contributors

- **Add a new classical denoiser**: See [`classic_denoisers/README.md`](classic_denoisers/README.md)
- **Add a new transform**: See [`transforms/README.md`](transforms/README.md)
- **Add a new ML approach**: Subclass `ImageTranslatorBase`, then create a corresponding `DenoiseRestorationBase` subclass in [`../restoration/denoise/`](../restoration/denoise/README.md)

## Related Packages

- [`../restoration/denoise/`](../restoration/denoise/README.md) — High-level denoiser wrappers that compose translators with transforms
- [`../features/`](../features/README.md) — Feature generation used by `ImageTranslatorFGR`
- [`../regression/`](../regression/README.md) — Regression backends used by `ImageTranslatorFGR`
- [`../nn/`](../nn/README.md) — Neural network models and training used by `ImageTranslatorCNNTorch`
