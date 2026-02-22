# Neural Networks (`aydin/nn/`)

This package provides the PyTorch neural network infrastructure for CNN-based image denoising in Aydin. It includes model architectures, training methods, dataset utilities, custom layers, and optimizers.

## Architecture

```
aydin/nn/
├── models/              # PyTorch model architectures
├── training_methods/    # Training loops (Noise2Self, Noise2Truth)
├── datasets/            # Dataset classes for patch extraction and masking
├── layers/              # Custom convolution and pooling layers
├── optimizers/          # Custom optimizers (ESAdam)
└── utils/               # Masking and center-smoothing utilities
```

### Integration

The neural network components are used by `ImageTranslatorCNNTorch` (`../it/cnn_torch.py`), which:
1. Selects a model architecture from `models/`
2. Extracts patches and creates datasets from `datasets/`
3. Trains using a method from `training_methods/`
4. Applies inference with tiling support from the base translator

## Subpackages

| Subpackage | Purpose | Details |
|------------|---------|---------|
| [`models/`](models/README.md) | Model architectures | UNet, JiNet, DnCNN, ResUNet, and variants |
| [`training_methods/`](training_methods/README.md) | Training loops | Noise2Self (self-supervised), Noise2Truth (supervised) |
| `datasets/` | Dataset classes | `GridMaskedDataset`, `RandomMaskedDataset`, `NoisyGroundtruthDataset`, `random_patches()` |
| `layers/` | Custom layers | `CustomConv`, `DilatedConv`, `PoolingDown`, `double_conv_block()` |
| `optimizers/` | Optimizers | `ESAdam` — Adam with decaying exploratory noise |
| `utils/` | Utilities | `Masking` wrapper, `apply_center_smoothing()` for JiNet post-processing |

## Key Patterns

### 2D and 3D Support

All models, layers, and datasets support both 2D and 3D images via a `spacetime_ndim` parameter that switches between `Conv2d`/`Conv3d`, `MaxPool2d`/`MaxPool3d`, etc.

### Self-Supervised Training

Noise2Self training uses grid-based pixel masking (4 patterns in 2D, 6 in 3D). The `GridMaskedDataset` handles masking with configurable replacement strategies: `'interpolate'` (default), `'zero'`, `'random'`, `'median'`.

### Blind-Spot CNN (JiNet)

`JINetModel` achieves J-invariance architecturally via dilated convolutions — the center pixel is excluded from the receptive field by construction, rather than by masking.

## For Contributors

- **Add a new model**: See [`models/README.md`](models/README.md)
- **Add a new training method**: See [`training_methods/README.md`](training_methods/README.md)
- **Add a new dataset type**: Subclass `torch.utils.data.Dataset` in `datasets/`

## Related Packages

- [`../it/`](../it/README.md) — `ImageTranslatorCNNTorch` uses this package for CNN-based denoising
- [`../restoration/denoise/`](../restoration/denoise/README.md) — `Noise2SelfCNN` wraps the CNN translator
