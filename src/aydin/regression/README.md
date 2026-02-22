# Regression (`aydin/regression/`)

This package provides pluggable regression backends for the Feature Generation & Regression (FGR) denoising approach. Each regressor learns a mapping from multi-scale image features to denoised pixel values.

## Available Regressors

| Module | Class | Backend | Notes |
|--------|-------|---------|-------|
| `cb.py` | `CBRegressor` | CatBoost | Default for FGR; best quality/speed balance |
| `lgbm.py` | `LGBMRegressor` | LightGBM | Faster training, slightly lower quality |
| `linear.py` | `LinearRegressor` | scikit-learn | Fastest; baseline for comparison |
| `perceptron.py` | `PerceptronRegressor` | PyTorch | Small neural network regressor |
| `random_forest.py` | `RandomForestRegressor` | scikit-learn | Ensemble method via random forests |
| `support_vector.py` | `SupportVectorRegressor` | scikit-learn | SVR for small-scale problems |

## Architecture

### `RegressorBase` (`base.py`)

Abstract base class providing:
- **Multi-channel support** — Trains one internal model per output channel
- **Early stopping** — Hooks for monitoring validation loss
- **Serialization** — `save()` / `load()` via JSON pickle
- **Loss tracking** — Per-channel loss history during training

### Subclass Interface

```python
class RegressorBase:
    def _fit(self, x_train, y_train, x_valid, y_valid, regressor_callback=None):
        """Train a single-channel model. Returns a fitted model object with a predict() method."""
        ...
```

The public `predict()` method is provided by the base class — it iterates over per-channel models and calls each model's `predict()` method. Subclasses only need to implement `_fit()`.

### Utility Subpackages

| Subpackage | Purpose |
|------------|---------|
| `cb_utils/` | `CatBoostStopTrainingCallback` — Custom early-stopping callback for CatBoost |
| `gbm_utils/` | LightGBM callbacks and `LightGBMModelAssembler` for model export |
| `nn_utils/` | PyTorch model definitions for the perceptron backend |

## Key Patterns

### Per-Channel Models

Each regressor trains a separate internal model per output channel. This enables natural multi-channel image support and allows different channels to converge independently.

### Pluggable Backend Selection

`Noise2SelfFGR` discovers available regressors dynamically and exposes them as variants (e.g., `'Noise2SelfFGR-cb'`, `'Noise2SelfFGR-lgbm'`). Users select backends via the variant string.

## Important: HTML Docstrings

Regressor class docstrings contain HTML tags rendered by the Aydin GUI:
- **`<a href="...">`** — GitHub links to library repositories (CatBoost, LightGBM, scikit-learn)
- **`<notgui>`** — Separates GUI-visible description from API-only sections

**Do not remove or convert these tags.** See [CLAUDE.md](../../CLAUDE.md) for full docstring conventions.

## For Contributors

To add a new regression backend:

1. Create `my_regressor.py` in this directory
2. Subclass `RegressorBase` and implement `_fit()` (return a model object with a `predict()` method)
3. Add `<a href>` and `<notgui>` tags in the class docstring
4. The regressor will be auto-discovered by `Noise2SelfFGR` as a new variant

## Related Packages

- [`../features/`](../features/README.md) — Feature generation that produces inputs for regression
- [`../it/`](../it/README.md) — `ImageTranslatorFGR` orchestrates features + regression
- [`../restoration/denoise/`](../restoration/denoise/README.md) — `Noise2SelfFGR` composes the full FGR pipeline
