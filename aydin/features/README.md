# Feature Engineering (`aydin/features/`)

This package generates multi-scale, multi-type image features used by the Feature Generation & Regression (FGR) denoising approach. Features encode spatial context around each pixel for regression-based denoising.

## Architecture

### Three-Tier Hierarchy

```
FeatureGeneratorBase (abstract)
    └── ExtensibleFeatureGenerator (composable)
        └── StandardFeatureGenerator (pre-configured, recommended)
```

1. **`FeatureGeneratorBase`** (`base.py`) — Abstract base defining `compute()`, `save()`, `load()`, and memory management (offcore/memmap support)
2. **`ExtensibleFeatureGenerator`** (`extensible_features.py`) — Composable generator that accumulates `FeatureGroupBase` instances via `add_feature_group()`
3. **`StandardFeatureGenerator`** (`standard_features.py`) — Pre-configured generator combining uniform, spatial, median, DCT, low-pass, and random features with 50+ tunable parameters. Recommended for most use cases.

### Feature Groups Subpackage

The [`groups/`](groups/README.md) subpackage contains 10 feature group implementations, each computing a different type of image feature. See [`groups/README.md`](groups/README.md) for details.

## Key Patterns

### Feature Computation

Features are computed as a stack of 2D arrays (one per feature) from an input image. Each feature group contributes one or more feature channels:

```python
generator = StandardFeatureGenerator()
features = generator.compute(image, exclude_center_feature=True, exclude_center_value=True)
# features shape: (num_features, *image_shape)
```

The `exclude_center_feature` and `exclude_center_value` parameters enable blind-spot (J-invariant) feature computation — the center pixel is excluded from each feature to enable self-supervised training.

### Integration with FGR Pipeline

```
Input Image → StandardFeatureGenerator.compute() → Feature Stack
Feature Stack + Target → RegressorBase.fit()
Feature Stack → RegressorBase.predict() → Denoised Image
```

This pipeline is orchestrated by `ImageTranslatorFGR` in [`../it/fgr.py`](../it/README.md).

## Important: HTML Docstrings

Feature generator class docstrings in `standard_features.py` and `extensible_features.py` contain:
- **`<notgui>`** — Separates GUI-visible description from API-only Attributes/Parameters sections

**Do not remove these tags.** See [CLAUDE.md](../../CLAUDE.md) for full docstring conventions.

## For Contributors

- **Add a new feature group**: See [`groups/README.md`](groups/README.md)
- **Create a custom feature generator**: Subclass `ExtensibleFeatureGenerator` and call `add_feature_group()` with your desired combination

## Related Packages

- [`../it/`](../it/README.md) — `ImageTranslatorFGR` orchestrates feature generation and regression
- [`../regression/`](../regression/README.md) — Regression backends that consume generated features
- [`../restoration/denoise/`](../restoration/denoise/README.md) — `Noise2SelfFGR` composes features + regression into a complete denoiser
