# Restoration (`aydin/restoration/`)

This package provides the high-level user-facing API for image restoration in Aydin. It wraps the lower-level Image Translator framework with sensible defaults, transform pipelines, and a unified interface used by both the GUI and CLI.

## Architecture

```
aydin/restoration/
├── __init__.py
└── denoise/           # Denoising restoration methods
    ├── base.py        # DenoiseRestorationBase (abstract)
    ├── classic.py     # Classical denoiser wrappers
    ├── noise2selffgr.py   # Feature Generation & Regression
    ├── noise2selfcnn.py   # CNN-based denoising
    └── util/          # Shared utilities (denoise_utils.py)
```

Currently, only the `denoise/` subdomain is implemented. The package structure allows future restoration types (e.g., super-resolution, inpainting) to be added as sibling subpackages.

## Key Patterns

### Two-Layer Abstraction

Aydin uses a two-layer design for denoising:

1. **High-level** (`restoration/denoise/`) — `DenoiseRestorationBase` subclasses compose translators with transforms, expose arguments for GUI/CLI, and handle model archiving
2. **Low-level** (`it/`) — `ImageTranslatorBase` subclasses implement the actual denoising algorithms

Users interact with the high-level layer; the low-level layer handles computation.

### Variant System

Each denoiser has a `variant` string that selects the specific algorithm:
- `Classic` variants: `'butterworth'`, `'gaussian'`, `'nlm'`, `'tv'`, `'wavelet'`, etc.
- `Noise2SelfFGR` variants: `'Noise2SelfFGR-cb'` (CatBoost), `'Noise2SelfFGR-lgbm'` (LightGBM), etc.
- `Noise2SelfCNN` variants: `'Noise2SelfCNN-unet'`, `'Noise2SelfCNN-jinet'`, etc.

### Discovery Mechanism

`DenoiseRestorationBase` provides class methods to discover available implementations and their arguments at runtime, which the GUI and CLI use to dynamically build their interfaces.

## For Contributors

See [`denoise/README.md`](denoise/README.md) for details on how to add new denoising approaches.

## Related Packages

- [`../it/`](../it/README.md) — Core Image Translator framework (low-level implementations)
- [`../gui/`](../gui/README.md) — GUI that builds its denoising UI from restoration class metadata
- [`../cli/`](../cli/README.md) — CLI that invokes restoration classes for command-line denoising
