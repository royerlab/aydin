# Command-Line Interface (`aydin/cli/`)

This package provides the Click-based CLI for Aydin, registered as the `aydin` console script entry point.

## Architecture

```
aydin/cli/
├── __init__.py
├── cli.py          # Click command group and all subcommands
└── styling.py      # Terminal output styling utilities
```

### Entry Point

The CLI is registered in `pyproject.toml` as:
```
[project.scripts]
aydin = "aydin.cli.cli:cli"
```

Running `aydin` with no arguments launches the GUI. Subcommands provide headless operations.

## Available Commands

| Command | Description |
|---------|-------------|
| `aydin` | Launch Aydin Studio GUI (default, no subcommand) |
| `aydin denoise <files>` | Denoise images using a specified algorithm and variant |
| `aydin info <file>` | Display image file metadata (shape, dtype, axes) |
| `aydin view <file>` | Open image in napari viewer |
| `aydin split_channels <file>` | Split multi-channel image into separate files |
| `aydin hyperstack <files>` | Combine multiple images into a hyperstack |
| `aydin ssim <a> <b>` | Compute SSIM between two images |
| `aydin psnr <a> <b>` | Compute PSNR between two images |
| `aydin mse <a> <b>` | Compute MSE between two images |
| `aydin fsc <a> <b>` | Compute Fourier Shell Correlation |
| `aydin benchmark_algos` | Benchmark available denoising algorithms |
| `aydin cite` | Print citation information |

### Key Options for `denoise`

- `--algorithm` / `-a` — Algorithm family (`Classic`, `Noise2SelfFGR`, `Noise2SelfCNN`)
- `--variant` / `-v` — Specific variant (e.g., `butterworth`, `cb`, `unet`)
- `--list-denoisers` — List all available algorithm/variant combinations
- File glob expansion is supported (e.g., `aydin denoise *.tif`)

## For Contributors

To add a new CLI subcommand:

1. Define a new Click command function in `cli.py`
2. Decorate with `@cli.command()`
3. Add Click options/arguments as needed
4. The command is automatically registered via the Click group

## Related Packages

- [`../restoration/denoise/`](../restoration/denoise/README.md) — Denoiser classes invoked by the `denoise` command
- [`../io/`](../io/README.md) — Image I/O used for reading/writing files
- [`../analysis/`](../analysis/README.md) — Analysis functions used by metric commands (ssim, psnr, fsc)
- [`../gui/`](../gui/README.md) — GUI launched when no subcommand is given
