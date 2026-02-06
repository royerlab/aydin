# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aydin is a self-supervised, auto-tuned image denoising tool for n-dimensional images. It provides three interfaces: GUI (Aydin Studio), CLI, and Python API. Supports classical denoisers, patch-based methods, and machine learning approaches (CNN, Gradient Boosting).

## Common Commands

### Development Setup
```bash
conda create -n aydin python=3.9
conda activate aydin
make setup  # or: pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests (excludes heavy, gpu, unstable by default)
make test

# Run a single test file
pytest aydin/path/to/test_file.py --disable-pytest-warnings

# Run a single test function
pytest aydin/path/to/test_file.py::test_function_name --disable-pytest-warnings

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
cd docs && make html        # Build HTML docs
cd docs && make publish     # Build multi-version docs
```

## Architecture

### Core Components

**Image Translator Framework** (`aydin/it/`):
- `base.py` - Core `ImageTranslatorBase` class that handles multi-dimensional arrays with batch/channel/spatial dimensions, training/inference slicing, transforms, and normalization
- `classic.py`, `cnn.py`, `fgr.py` - Implementations wrapping different denoising approaches

**Denoising Algorithms** (`aydin/restoration/denoise/`):
- `classic.py` - Classical denoisers (Butterworth, Gaussian, NLM, Total Variation)
- `noise2selffgr.py` - Self-supervised Feature Generation & Regression (recommended approach)
- `noise2selfcnn.py` - Self-supervised CNN approach

**Regression Methods** (`aydin/regression/`):
- `cb.py` - CatBoost (default for FGR)
- `lgbm.py` - LightGBM
- Other: linear, perceptron, random forest, SVM

**Feature Engineering** (`aydin/features/`):
- `standard_features.py`, `extensible_features.py` - Feature generation for ML denoisers

**Transforms** (`aydin/it/transforms/`):
- Image preprocessing/postprocessing: deskew, motion correction, high-pass, range normalization, histogram stretching, VST, padding

**I/O** (`aydin/io/`):
- `io.py` - Image reading/writing supporting TIFF, CZI, ND2, Zarr formats

**Interfaces**:
- `aydin/gui/` - PyQt5-based GUI (Aydin Studio)
- `aydin/cli/cli.py` - Click-based CLI

### Test Markers

Tests use pytest markers defined in `pyproject.toml`:
- `@pytest.mark.heavy` - Long-running tests
- `@pytest.mark.gpu` - Requires NVIDIA GPU
- `@pytest.mark.unstable` - Flaky tests

### Logging

Use the internal logging API at `aydin/util/log/log.py` instead of print statements.

## Code Style Guidelines

- Follow PEP8 + Black formatter (config in pyproject.toml, single quotes preserved)
- Use NumPy-style docstrings
- Flake8 ignores: E501 (line length), E203, E741, W503
