# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aydin is a self-supervised, auto-tuned image denoising tool for n-dimensional images. It provides four interfaces: GUI (Aydin Studio), napari plugin, CLI, and Python API. Supports classical denoisers, patch-based methods, and machine learning approaches (CNN, Gradient Boosting).

## Common Commands

### Development Setup
```bash
pip install -e ".[dev]"
# Or use the Makefile (also installs docs deps + pre-commit hooks):
make setup
```

### Testing
```bash
# Run all tests (excludes heavy, gpu, unstable by default)
make test

# Run a single test file
pytest src/aydin/path/to/test_file.py --disable-pytest-warnings

# Run a single test function
pytest src/aydin/path/to/test_file.py::test_function_name --disable-pytest-warnings

# Run heavy tests only (marked @pytest.mark.heavy)
make test-heavy

# Run GPU tests only (marked @pytest.mark.gpu)
make test-gpu
```

### Code Style
```bash
# Format code (isort + black)
make format

# Check formatting and lint
make check
```

### Running the Application
```bash
aydin                       # Launch GUI (Aydin Studio)
aydin denoise image.tif     # CLI denoising
aydin -h                    # Help
```

### Documentation
```bash
make docs                   # Build HTML docs (current version)
make docs-build             # Build multi-version docs (all tags)
make docs-publish           # Build and deploy to GitHub Pages
```

## Architecture

### Project Layout

Uses a **`src/` layout**: source code lives in `src/aydin/`. This prevents accidental imports of the uninstalled package from the repo root.

### Core Components

**Image Translator Framework** (`src/aydin/it/`):
- `base.py` - Core `ImageTranslatorBase` class that handles multi-dimensional arrays with batch/channel/spatial dimensions, training/inference slicing, transforms, and normalization
- `classic.py`, `cnn_torch.py`, `fgr.py` - Implementations wrapping different denoising approaches

**Denoising Algorithms** (`src/aydin/restoration/denoise/`):
- `classic.py` - Classical denoisers (Butterworth, Gaussian, NLM, Total Variation)
- `noise2selffgr.py` - Self-supervised Feature Generation & Regression (recommended approach)
- `noise2selfcnn.py` - Self-supervised CNN approach

**Regression Methods** (`src/aydin/regression/`):
- `cb.py` - CatBoost (default for FGR)
- `lgbm.py` - LightGBM
- Other: linear, perceptron, random forest, SVM

**Feature Engineering** (`src/aydin/features/`):
- `standard_features.py`, `extensible_features.py` - Feature generation for ML denoisers

**Transforms** (`src/aydin/it/transforms/`):
- Image preprocessing/postprocessing: deskew, motion correction, high-pass, range normalization, histogram stretching, VST, padding

**I/O** (`src/aydin/io/`):
- `io.py` - Image reading/writing supporting TIFF, CZI, ND2, Zarr formats

**Interfaces**:
- `src/aydin/gui/` - PyQt6-based GUI (Aydin Studio)
- `src/aydin/cli/cli.py` - Click-based CLI

### Build System

- **Build backend**: hatchling (configured in `pyproject.toml`)
- **Version**: Single source of truth is `src/aydin/__init__.py` (`__version__`). `pyproject.toml` reads it dynamically via `[tool.hatch.version]`.
- **Makefile**: Orchestrates common commands (`make help` for full list)
  - `make setup` / `make install-dev` - Install for development
  - `make test` / `make test-heavy` / `make test-gpu` - Run tests
  - `make format` / `make check` - Code formatting and linting
  - `make validate` - Pre-publish checks (format + lint + clean tree)
  - `make build` / `make clean` - Build and clean artifacts
  - `make publish` / `make publish-patch` - Create release PR (see Release Process below)

### Test Markers

Tests use pytest markers defined in `pyproject.toml`:
- `@pytest.mark.heavy` - Long-running tests
- `@pytest.mark.gpu` - Requires NVIDIA GPU
- `@pytest.mark.unstable` - Flaky tests

### Logging

Use the internal logging API at `src/aydin/util/log/log.py` instead of print statements.

### Versioning

The project uses **calendar versioning**: `YYYY.M.D` (e.g., `2025.2.4`), with an optional `.patch` suffix for same-day releases (e.g., `2025.2.4.1`).

- **Single source of truth**: `src/aydin/__init__.py` → `__version__ = "YYYY.M.D"`
- `pyproject.toml` reads it dynamically via `[tool.hatch.version]` — never edit the version there
- `docs/source/conf.py` also reads from `__init__.py` at build time

### Release Process

Releases go through a PR-based flow. Requires the GitHub CLI (`gh`).

```bash
# Regular release (bumps to today's date, e.g., 2025.2.16)
make publish

# Patch release (e.g., 2025.2.16.1 → 2025.2.16.2)
make publish-patch
```

**What `make publish` does:**
1. Runs `make validate` (checks: on main, clean tree, formatting, lint)
2. Creates a `release/vYYYY.M.D` branch
3. Bumps `__version__` in `src/aydin/__init__.py`
4. Pushes and creates a PR via `gh pr create`
5. Switches back to `main`

**After that, you merge the PR on GitHub.** Then the automated pipeline takes over:

```
PR merged → release.yml creates git tag → publish.yml: verify → test → build → PyPI → GitHub Release
```

### CI/CD Pipeline

**Workflows** (`.github/workflows/`):

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `test_pull_requests.yml` | PR to `main` | Style, lint, tests across Python 3.9–3.13, Linux/macOS/Windows |
| `release.yml` | Release PR merged | Auto-creates git tag, calls `publish.yml` |
| `publish.yml` | Tag push / workflow_call / manual | Verify tag, test, build, publish to PyPI, create GitHub Release |
| `labeler.yml` | PR opened | Auto-labels PRs by changed paths |
| `link_check.yml` | PR to `main` | Checks Markdown links |

**Why `release.yml` calls `publish.yml` directly**: Git tags pushed by `GITHUB_TOKEN` don't trigger other workflows (GitHub security restriction). So `release.yml` uses `workflow_call` to invoke `publish.yml` directly instead of relying on the tag push event. The `push: tags` trigger on `publish.yml` remains as a fallback for manual tag pushes. A `workflow_dispatch` trigger is also available for manual retries from the GitHub UI.

**PyPI authentication**: Uses OIDC trusted publishing (no API tokens). The `publish-to-pypi` job runs in the `pypi` environment with `id-token: write` permission.

### Linting Configuration

- **Config file**: `.flake8` (repo root) — single source of truth for flake8 settings
- Used by: Makefile (`make lint`), CI lint job, and pre-commit hook
- Per-file `E501` ignores are set for GUI files with HTML-containing docstrings

## Code Style Guidelines

- Follow PEP8 + Black formatter (config in pyproject.toml, single quotes preserved)
- Use NumPy-style docstrings
- Flake8 config in `.flake8` (ignores E203, E741, W503, and others; per-file E501 for GUI docstrings)

## GUI Docstring Conventions (CRITICAL)

**The Aydin GUI (Aydin Studio) renders Python docstrings as HTML using Qt's RichText.** Docstrings in certain modules contain intentional HTML markup that MUST be preserved. Removing or converting these tags breaks the GUI display.

### HTML Tags Used in Docstrings

1. **`<a href="...">text</a>`** - Clickable hyperlinks rendered in the GUI. Found in:
   - `src/aydin/it/classic_denoisers/*.py` - Wikipedia links for algorithm names (Butterworth, NLM, Wavelet, Lipschitz, Bilateral, BM3D/ND, Harmonic, TV)
   - `src/aydin/it/transforms/highpass.py` - Wikipedia link for 'blue' noise
   - `src/aydin/restoration/denoise/noise2self*.py` - arXiv link to Noise2Self paper
   - `src/aydin/regression/cb.py`, `lgbm.py`, `random_forest.py` - GitHub links to libraries
   - `src/aydin/gui/tabs/qt/summary.py` - DOI, GitHub, and forum links

2. **`<moreless>`** - Custom tag that creates an expandable "Read more/Read less" UI section. Used in GUI tab class docstrings:
   - `src/aydin/gui/tabs/qt/denoise.py`
   - `src/aydin/gui/tabs/qt/processing.py`
   - `src/aydin/gui/tabs/qt/dimensions.py`
   - `src/aydin/gui/tabs/qt/summary.py`
   - `src/aydin/gui/tabs/qt/base_cropping.py`

3. **`<split>`** - Used with `<moreless>` to create a two-column layout in the expanded section.

4. **`<br>` / `<br><br>`** - HTML line breaks. Also, literal `\n\n` in docstrings is converted to `<br><br>` at runtime by the restoration modules.

5. **`\n\n`** (literal backslash-n) - Used as explicit paragraph break markers in classic denoiser docstrings. The GUI code converts these to `<br><br>` via `.replace("\n\n", "<br><br>")`.

6. **`<notgui>`** - Marks the boundary between GUI-visible content and API-only content. Everything before this tag is shown in the GUI; everything after (Parameters, Attributes, etc.) is only visible via `help()`, Sphinx, and IDEs. Stripped at runtime by `strip_notgui()` in `src/aydin/util/string/break_text.py`. Used in:
   - `src/aydin/it/classic_denoisers/*.py` - In `denoise_*` function docstrings, before `Parameters`
   - `src/aydin/it/transforms/*.py` - In transform class docstrings
   - `src/aydin/regression/*.py` - In regressor class docstrings
   - `src/aydin/nn/models/*.py` - In model class docstrings
   - `src/aydin/features/standard_features.py`, `extensible_features.py` - In feature generator class docstrings
   - `src/aydin/it/classic.py`, `src/aydin/restoration/denoise/noise2self*.py` - In denoiser class docstrings

### How Docstrings Flow to the GUI

- **Classic denoisers**: `src/aydin/restoration/denoise/classic.py` reads `denoise_*.__doc__`, applies `strip_notgui()`, and converts `\n\n` to `<br><br>`
- **FGR regressors**: `src/aydin/restoration/denoise/noise2selffgr.py` reads regressor class `__doc__`, applies `strip_notgui()`, and converts `\n\n` to `<br><br>`
- **CNN models**: `src/aydin/restoration/denoise/noise2selfcnn.py` reads model class `__doc__`, applies `strip_notgui()`, and converts `\n\n` to `<br><br>`
- **Transforms**: `src/aydin/gui/_qt/transforms_tab_item.py` reads transform class `__doc__`, applies `strip_notgui()`, and converts `\n` to `<br>`
- **Tab descriptions**: `QReadMoreLessLabel` widget parses `<moreless>` and `<split>` markers

### Rules When Editing Docstrings

- **NEVER remove `<a href>` links** from docstrings in the modules listed above
- **NEVER remove `<moreless>`, `<split>`, `<br>`, or `<notgui>` tags** from GUI tab docstrings
- **NEVER convert HTML links to RST/Markdown format** - the GUI needs raw HTML
- **Preserve literal `\n\n`** paragraph markers in classic denoiser docstrings
- When improving docstrings, keep the HTML tags intact and only modify the surrounding text
- **Use `<notgui>`** to separate GUI-visible content from API-only sections (Attributes, Parameters, etc.). Place `<notgui>` on its own line before any section you want hidden from the GUI
