# Analysis (`aydin/analysis/`)

This package provides image quality metrics, noise characterization, and dimensional analysis tools used internally by Aydin and exposed via the CLI.

## Modules

### Quality Metrics

| Module | Function | Description |
|--------|----------|-------------|
| `image_metrics.py` | Various | SSIM, PSNR, mutual information, and other image comparison metrics |
| `fsc.py` | `fsc()` | Fourier Shell Correlation — resolution-independent quality assessment |

### Noise Characterization

| Module | Function | Description |
|--------|----------|-------------|
| `blind_spot_analysis.py` | — | Detects optimal blind-spot patterns for self-supervised denoising |
| `empirical_noise_model.py` | — | Estimates noise statistics empirically from a single image |
| `snr_estimate.py` | — | Signal-to-noise ratio estimation |
| `camera_simulation.py` | — | Simulates camera noise models (Poisson, Gaussian, mixed) |

### Image Analysis

| Module | Function | Description |
|--------|----------|-------------|
| `dimension_analysis.py` | — | Classifies axes as batch, channel, or spatio-temporal |
| `correlation.py` | — | Spatial correlation analysis |
| `resolution_estimate.py` | — | Resolution estimation from image content |
| `find_kernel.py` | — | Blur kernel (PSF) estimation |

## Usage

Quality metrics are used by:
- **CLI**: `aydin ssim`, `aydin psnr`, `aydin mse`, `aydin fsc` commands
- **Internally**: Training quality monitoring and auto-calibration in the IT framework
- **Dimension analysis**: Used by the GUI's Dimensions tab and `ImageTranslatorBase` to interpret array axes

## For Contributors

To add a new analysis function:

1. Create a module in this directory
2. Implement the function with a clear NumPy-style docstring
3. If it's a quality metric, consider adding a CLI command in [`../cli/`](../cli/README.md)

## Related Packages

- [`../cli/`](../cli/README.md) — CLI exposes metrics as subcommands
- [`../it/`](../it/README.md) — IT framework uses dimension analysis and quality metrics internally
- [`../gui/`](../gui/README.md) — GUI uses dimension analysis in the Dimensions tab
