# Feature Groups (`aydin/features/groups/`)

This subpackage contains the individual feature group implementations used by `ExtensibleFeatureGenerator` and `StandardFeatureGenerator`.

## Available Feature Groups

| Module | Class | Description |
|--------|-------|-------------|
| `uniform.py` | `UniformFeatures` | Multi-scale box (integral) filter features; CUDA-accelerated for large images |
| `spatial.py` | `SpatialFeatures` | Position-dependent coordinate features; learns illumination gradients |
| `median.py` | `MedianFeatures` | Robust median filter features at multiple radii |
| `lowpass.py` | `LowPassFeatures` | Butterworth low-pass impulse-response kernels at multiple cutoff frequencies |
| `dct.py` | `DCTFeatures` | Discrete Cosine Transform basis function features |
| `random.py` | `RandomFeatures` | Deterministic random convolutional filter features |
| `correlation.py` | `CorrelationFeatures` | Generic kernel-based correlation features (base for lowpass, dct, random, learned_conv) |
| `learned_conv.py` | `LearnedCorrelationFeatures` | Data-driven kernels learned via MiniBatchKMeans clustering |
| `translations.py` | `TranslationFeatures` | Shifted image copies as features (simplest spatial context) |
| `extract_kernels.py` | `extract_kernels()` | Utility function to extract representative patterns via clustering |

### Inheritance

```
FeatureGroupBase (abstract, base.py)
├── UniformFeatures
├── SpatialFeatures
├── MedianFeatures
├── TranslationFeatures
└── CorrelationFeatures
    ├── LowPassFeatures
    ├── DCTFeatures
    ├── RandomFeatures
    └── LearnedCorrelationFeatures
```

## Key Patterns

### `FeatureGroupBase` Interface

All feature groups implement:

```python
class FeatureGroupBase:
    @property
    def receptive_field_radius(self) -> int: ...
    def num_features(self, ndim: int) -> int: ...
    def prepare(self, image, excluded_voxels=None, **kwargs): ...
    def compute_feature(self, index: int, feature): ...
```

- `receptive_field_radius` — Maximum spatial extent of the features (in pixels)
- `num_features(ndim)` — Number of feature channels this group produces for a given dimensionality
- `prepare(image, excluded_voxels)` — Pre-computation step (e.g., building kernels, computing filtered images)
- `compute_feature(index, feature)` — Compute a single feature by index, storing the result in-place into the pre-allocated `feature` array

### J-Invariance Support

All groups support `excluded_voxels` — a specification of which pixels to exclude from feature computation. This enables blind-spot denoising where the center pixel's value is never used as input, ensuring self-supervised training validity.

## For Contributors

To add a new feature group:

1. Create `my_features.py` in this directory
2. Subclass `FeatureGroupBase` (or `CorrelationFeatures` for kernel-based features)
3. Implement `receptive_field_radius`, `num_features()`, `prepare()`, and `compute_feature(index, feature)`
4. Register the group in `StandardFeatureGenerator` or use it via `ExtensibleFeatureGenerator.add_feature_group()`

## Related Packages

- [`../`](../README.md) — Parent feature engineering package (generators that compose feature groups)
- [`../../it/`](../../it/README.md) — `ImageTranslatorFGR` orchestrates feature computation
