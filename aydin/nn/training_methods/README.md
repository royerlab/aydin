# Training Methods (`aydin/nn/training_methods/`)

This subpackage contains training loops for CNN-based denoising in Aydin.

## Available Methods

| Module | Function | Approach | Supervision |
|--------|----------|----------|-------------|
| `n2s.py` | `n2s_train()` | Noise2Self | Self-supervised (single noisy image) |
| `n2t.py` | `n2t_train()` | Noise2Truth | Supervised (paired noisy/clean images) |
| `n2s_shiftconv.py` | `n2s_shiftconv_train()` | Shift-Convolution N2S | Self-supervised with shift-based J-invariance |

## Training Details

### `n2s_train()` — Noise2Self

- **Input**: Single noisy image + model
- **Masking**: Grid-based pixel masking (4 patterns in 2D, 6 in 3D) via `GridMaskedDataset`
- **Loss**: MSE
- **Optimizer**: AdamW
- **Scheduling**: ReduceLROnPlateau with configurable patience
- **Early stopping**: Best-model checkpointing with patience-based termination

### `n2t_train()` — Noise2Truth

- **Input**: Paired noisy and clean images + model
- **Dataset**: `NoisyGroundtruthDataset` with optional training noise injection
- **Loss**: L1
- **Optimizer**: ESAdam (Adam with exploratory stochastic noise)
- **Regularization**: L2 weight regularization
- **Logging**: TensorBoard support

### `n2s_shiftconv_train()` — Shift-Convolution

- **Input**: Single noisy image + model
- **Approach**: Rotates images (4 ways in 2D, 6 in 3D), applies column/row shifts at each convolution to exclude center pixel
- **J-invariance**: Achieved via architectural shifts rather than pixel masking

## For Contributors

To add a new training method:

1. Create `my_method.py` in this directory
2. Implement a `my_method_train(input_image, model, **kwargs)` function
3. Use datasets from `../datasets/` or create new ones
4. Include early stopping and best-model checkpointing
5. Wire it into `ImageTranslatorCNNTorch` in `../../it/cnn_torch.py`

## Related Packages

- [`../models/`](../models/README.md) — Model architectures trained by these methods
- [`../datasets/`](../datasets/) — Dataset classes for patch extraction and masking
- [`../optimizers/`](../optimizers/) — ESAdam optimizer used by Noise2Truth
- [`../../it/`](../../it/README.md) — `ImageTranslatorCNNTorch` selects and runs training methods
