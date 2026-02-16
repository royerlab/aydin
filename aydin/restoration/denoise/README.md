# Denoising Restoration (`aydin/restoration/denoise/`)

This subpackage contains the three main denoising approaches in Aydin, each wrapping a different Image Translator with appropriate transforms and defaults.

## Implementations

### `DenoiseRestorationBase` (`base.py`)

Abstract base class providing:
- Implementation discovery (`get_implementations_in_a_module()`)
- Argument inspection for GUI/CLI integration
- Model archiving (save/load trained denoisers)
- Variant resolution (mapping variant strings to concrete classes)

### `Classic` (`classic.py`)

Wraps `ImageDenoiserClassic` with padding, range normalization, and variance stabilization transforms. Exposes 14 classical algorithms as variants — each auto-calibrated via J-invariance.

Available variants: `butterworth`, `gaussian`, `gm`, `nlm`, `spectral`, `wavelet`, `harmonic`, `tv`, `bilateral`, `lipschitz`, `pca`, `bmnd`, `dictionary_fixed`, `dictionary_learned`

### `Noise2SelfFGR` (`noise2selffgr.py`) — Recommended

Wraps `ImageTranslatorFGR` with `StandardFeatureGenerator` and a pluggable regression backend. This is the recommended approach for most images — it is fast, memory-efficient, and produces high-quality results.

Available variants: `Noise2SelfFGR-cb` (CatBoost, default), `Noise2SelfFGR-lgbm`, `Noise2SelfFGR-linear`, `Noise2SelfFGR-perceptron`, `Noise2SelfFGR-random_forest`, `Noise2SelfFGR-support_vector`

### `Noise2SelfCNN` (`noise2selfcnn.py`)

Wraps `ImageTranslatorCNNTorch` with self-supervised CNN architectures. Best for images with complex spatial structure where CNN inductive biases help.

Available variants: `Noise2SelfCNN-unet`, `Noise2SelfCNN-jinet`

## How Arguments Flow to GUI/CLI

Each denoiser class defines its constructor parameters with type annotations and defaults. The `DenoiseRestorationBase` discovery mechanism inspects these to:
1. Generate GUI widgets (sliders, dropdowns, checkboxes)
2. Generate CLI options (`--variant`, `--max-epochs`, etc.)
3. Validate user inputs at runtime

## Important: HTML Docstrings

The class docstrings in this package contain HTML tags that are rendered by the GUI:
- `<a href="...">` links (e.g., arXiv paper references)
- `<notgui>` tags separating GUI-visible from API-only content
- `\n\n` paragraph breaks converted to `<br><br>` at runtime

See the project [CLAUDE.md](../../../CLAUDE.md) for full docstring conventions.

## For Contributors

To add a new denoising approach:

1. Create a new `ImageTranslatorBase` subclass in [`../../it/`](../../it/README.md) implementing `_train()` and `_translate()`
2. Create a new `DenoiseRestorationBase` subclass here that composes the translator with appropriate transforms
3. Register the class so it is discoverable by the variant system

## Related Packages

- [`../`](../README.md) — Parent restoration package
- [`../../it/`](../../it/README.md) — Image Translator implementations (the actual algorithms)
- [`../../it/classic_denoisers/`](../../it/classic_denoisers/README.md) — Classical denoiser functions used by `Classic`
- [`../../features/`](../../features/README.md) — Feature generation used by `Noise2SelfFGR`
- [`../../regression/`](../../regression/README.md) — Regression backends used by `Noise2SelfFGR`
- [`../../nn/`](../../nn/README.md) — Neural network models used by `Noise2SelfCNN`
