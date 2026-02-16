# Neural Network Models (`aydin/nn/models/`)

This subpackage contains PyTorch model architectures for CNN-based image denoising.

## Available Models

| Module | Class | Architecture | Key Feature |
|--------|-------|-------------|-------------|
| `unet.py` | `UNetModel` | Standard U-Net | Skip connections with concatenation; exponential filter growth (8 → 16 → 32...) |
| `jinet.py` | `JINetModel` | J-Invariant Net | Blind-spot CNN via dilated convolutions; center pixel excluded by construction |
| `dncnn.py` | `DnCNNModel` | DnCNN | Feed-forward with batch normalization; no pooling or skip connections |
| `res_unet.py` | `ResidualUNetModel` | Residual U-Net | Additive skip connections (vs concatenation); reduced parameters |
| `linear_scaling_unet.py` | `LinearScalingUNetModel` | Linear-Scaling U-Net | Linear filter growth (8 → 16 → 24...); parameter-efficient |
| `ronneberger_unet.py` | `RonnebergerUNetModel` | Original U-Net | Ronneberger et al. (2015) architecture adapted for denoising |

All models extend `torch.nn.Module` and support both 2D and 3D inputs via `spacetime_ndim`.

## Key Patterns

### Common Constructor Parameters

```python
model = UNetModel(
    spacetime_ndim=2,      # 2 for 2D images, 3 for 3D volumes
    nb_unet_levels=3,      # Encoder/decoder depth
    nb_filters=8,          # Base filter count
    pooling_mode='max',    # 'max' or 'ave' pooling
)
```

### UNet vs JiNet

- **UNet** is a general-purpose encoder-decoder trained with Noise2Self pixel masking
- **JiNet** achieves J-invariance architecturally — dilated convolutions with increasing receptive fields ensure the center pixel is never in the receptive field, enabling self-supervised training without masking

### Demo Configurations

The `demo/` subdirectory contains pre-configured model + training combinations:
- `unet/n2s_2D_generic.py` — UNet with Noise2Self for 2D
- `unet/n2t_2D_generic.py` — UNet with Noise2Truth for 2D
- `jinet/n2s_2D_generic.py` — JiNet with Noise2Self for 2D
- `jinet/n2t_2D_generic.py` — JiNet with Noise2Truth for 2D

## Important: HTML Docstrings

Model class docstrings contain HTML tags rendered by the Aydin GUI:
- **`<notgui>`** — Separates GUI-visible description from API-only sections

**Do not remove these tags.** See [CLAUDE.md](../../../CLAUDE.md) for full docstring conventions.

## For Contributors

To add a new model architecture:

1. Create `my_model.py` in this directory
2. Subclass `torch.nn.Module` with a `spacetime_ndim` parameter
3. Use layers from `../layers/` (`CustomConv`, `DilatedConv`, `PoolingDown`) for 2D/3D compatibility
4. Add `<notgui>` tag in the class docstring before Parameters
5. Add demo configurations in `demo/` if appropriate

## Related Packages

- [`../training_methods/`](../training_methods/README.md) — Training loops that train these models
- [`../layers/`](../layers/) — Custom convolution and pooling layers used by models
- [`../../it/`](../../it/README.md) — `ImageTranslatorCNNTorch` selects and manages model instances
