# Conda-Constructor Packaging

This directory contains everything needed to build native OS installers for **Aydin Studio** using [conda-constructor](https://github.com/conda/constructor).

| Platform | Format | Output |
|----------|--------|--------|
| Linux x86_64 | `.sh` | Shell script installer |
| macOS ARM64 | `.pkg` | Native macOS GUI installer |
| Windows x64 | `.exe` | NSIS-based GUI installer |

## Architecture

The installer creates a single conda environment containing Python, Aydin, and all dependencies. A desktop shortcut ("Aydin Studio") is created via [menuinst](https://github.com/conda/menuinst) v2.

```
packaging/
  build_installer.py           # Generates construct.yaml and invokes constructor
  LICENSE.rtf                  # License for installer UI (macOS/Windows)
  conda-recipe/
    aydin/recipe.yaml          # Conda recipe for the aydin package
    aydin-menu/recipe.yaml     # Conda recipe for the desktop shortcut package
  menuinst/
    aydin-studio.json          # menuinst v2 JSON (cross-platform shortcuts)
  icons/
    aydin_icon.png             # Linux icon (295x295)
    aydin_icon.ico             # Windows icon (multi-size)
    aydin_icon.icns            # macOS icon
  scripts/
    generate_icons.py          # Converts source PNG to ICO/ICNS
  post_install/
    post_install.sh            # Unix post-install script
    post_install.bat           # Windows post-install script
  environments/
    build_installer.yml        # Conda env for building installers
```

## Prerequisites

1. **conda or mamba** installed (e.g., via [Miniforge](https://github.com/conda-forge/miniforge))
2. The `aydin` and `aydin-menu` packages must be available on a conda channel (conda-forge or a custom channel like `anaconda.org/royerlab`)

## Quick Start

```bash
# 1. Create the build environment (one-time setup)
make installer-env

# 2. Activate the environment
conda activate aydin-build-installer

# 3. Build the installer for your platform
make installer

# Output appears in _work/
ls _work/AydinStudio-*
```

## Dry Run

Generate `construct.yaml` without building:

```bash
python packaging/build_installer.py --dry-run
```

## Custom Channels

To include a custom channel (e.g., during bootstrap before conda-forge has the package):

```bash
python packaging/build_installer.py --channels conda-forge royerlab
```

## GPU Support

The installer ships **CPU-only PyTorch** to keep the download size reasonable. Users who need GPU acceleration can add CUDA support after installation:

```bash
# Activate the installed environment
conda activate /path/to/aydin

# Add CUDA support
conda install pytorch-cuda -c conda-forge
```

## Icon Generation

Icons are pre-committed to `packaging/icons/`. To regenerate from the source PNG:

```bash
# Requires Pillow: pip install Pillow
make installer-icons
```

On macOS, the script uses `iconutil` (from Xcode Command Line Tools) for best `.icns` quality. On other platforms, it falls back to Pillow's built-in `.icns` support.

## CI/CD

The GitHub Actions workflow (`.github/workflows/build_installers.yml`) builds installers for all 3 platforms.

- **On PRs** touching `packaging/`: runs a **dry-run** only (validates construct.yaml generation)
- **On `workflow_dispatch`**: builds real installers and uploads as artifacts
- **With a tag ref** (e.g., `v2026.2.18`): also attaches installers to the GitHub Release

### Manual Trigger

Go to Actions > "Build Installers" > Run workflow, and provide:
- **ref**: git tag or branch (e.g., `v2026.2.18`)
- **channels**: conda channels (default: `conda-forge`)

## Code Signing (Future)

Code signing is not yet enabled but the infrastructure is ready. When certificates are available:

### macOS

Set these GitHub Actions secrets:
- `CONSTRUCTOR_SIGNING_IDENTITY`: Developer ID Installer certificate name
- `CONSTRUCTOR_NOTARIZATION_IDENTITY`: Apple ID for notarization

The `build_installer.py` script reads these from environment variables.

### Windows

Set this secret:
- `CONSTRUCTOR_SIGNING_CERTIFICATE`: Path to PFX certificate file

Without code signing, installers work but users see OS security warnings (Gatekeeper on macOS, SmartScreen on Windows).

## Troubleshooting

### "constructor not found"

Activate the build environment: `conda activate aydin-build-installer`

### Solver conflicts

If conda's solver takes too long or fails:
1. Ensure `conda-libmamba-solver` is installed (included in `build_installer.yml`)
2. Try pinning fewer packages — `build_installer.py` lists explicit deps for solver reliability, but you can remove some if they cause conflicts

### Icon generation fails

Ensure Pillow is installed: `pip install Pillow`. For best `.icns` quality on macOS, install Xcode Command Line Tools: `xcode-select --install`.
