# Image Transforms (`aydin/it/transforms/`)

This subpackage provides 11 image preprocessing/postprocessing transforms that are applied before and after denoising. Transforms are composable and reversible — they are applied in order during preprocessing (`preprocess()`) and in reverse order during postprocessing (`postprocess()`).

## Available Transforms

| Module | Transform | Purpose |
|--------|-----------|---------|
| `range.py` | `RangeTransform` | Normalizes pixel values to [0, 1] |
| `variance_stabilisation.py` | `VarianceStabilisationTransform` | Anscombe transform to stabilize Poisson noise variance |
| `padding.py` | `PaddingTransform` | Pads image borders to reduce edge artifacts |
| `highpass.py` | `HighpassTransform` | Removes low-frequency content before denoising |
| `histogram.py` | `HistogramEqualisationTransform` | Histogram equalization / CLAHE |
| `deskew.py` | `DeskewTransform` | Corrects integral shear in light-sheet data |
| `motion.py` | `MotionStabilisationTransform` | Phase-correlation-based motion correction |
| `fixedpattern.py` | `FixedPatternTransform` | Suppresses axis-aligned fixed offset patterns |
| `periodic.py` | `PeriodicNoiseSuppressionTransform` | Removes periodic noise via FFT |
| `salt_pepper.py` | `SaltPepperTransform` | Corrects impulse (salt-and-pepper) noise |
| `attenuation.py` | `AttenuationTransform` | Corrects axis-aligned intensity attenuation |

## Key Patterns

### `ImageTransformBase` Interface (`base.py`)

All transforms implement this interface:

```python
class ImageTransformBase:
    def preprocess(self, array):
        """Preprocess image before denoising."""
        ...

    def postprocess(self, array):
        """Reverse preprocessing after denoising."""
        ...
```

Transforms store any state needed for inversion (e.g., original range, padding amounts) and support serialization for model persistence.

### Transform Pipeline

`ImageTranslatorBase` maintains an ordered list of transforms. During training and inference:
1. **Preprocessing**: Each transform's `preprocess()` is called in order
2. **Denoising**: The core algorithm operates on the transformed image
3. **Postprocessing**: Each transform's `postprocess()` is called in reverse order

### Default Transform Stacks

Each `DenoiseRestorationBase` subclass configures its own transform stack:
- **Classic**: Padding + Range + VarianceStabilisation
- **Noise2SelfFGR**: Range + VarianceStabilisation + Highpass (configurable)
- **Noise2SelfCNN**: Range + VarianceStabilisation (configurable)

## Important: HTML Docstrings

Transform class docstrings contain HTML tags rendered by the Aydin GUI:
- **`<a href="...">`** — Reference links (e.g., Wikipedia for "blue" noise in `highpass.py`)
- **`<notgui>`** — Separates GUI-visible description from API-only Attributes/Parameters sections

The GUI reads these docstrings via `transforms_tab_item.py` and converts `\n` to `<br>`. **Do not remove or convert these tags.**

## For Contributors

To add a new transform:

1. Create `my_transform.py` in this directory
2. Subclass `ImageTransformBase` and implement `preprocess()` / `postprocess()`
3. Add `<notgui>` tag before the Parameters/Attributes section in the docstring
4. Register the transform in `__init__.py`
5. Add it to the appropriate default transform stack in `restoration/denoise/`

## Related Packages

- [`../`](../README.md) — Parent Image Translator framework (manages transform pipelines)
- [`../../restoration/denoise/`](../../restoration/denoise/README.md) — Configures default transform stacks per denoiser
- [`../../gui/_qt/`](../../gui/_qt/README.md) — `transforms_tab_item.py` and `transforms_tab_widget.py` render transform UIs
