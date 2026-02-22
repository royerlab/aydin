# Napari Plugin (`aydin/napari_plugin/`)

This package provides the Aydin napari plugin, registered via the `napari.manifest` entry point in `pyproject.toml`.

## Architecture

```
aydin/napari_plugin/
├── napari.yaml          # npe2 manifest — declares widgets, samples, and activation hook
├── _widget.py           # AydinDenoiseWidget — dock widget with 8 denoising methods
├── _studio_bridge.py    # AydinStudioWidget — launches Aydin Studio from napari
├── _context_actions.py  # Right-click "Denoise (high quality)" / "Denoise (fast)" actions
├── _sample_data.py      # 18 sample datasets (2 synthetic + 16 Zenodo-hosted)
├── _axes_utils.py       # Napari axis metadata → Aydin FileMetadata bridge
└── tests/               # Widget and axes utility tests
```

### Entry Point

Registered in `pyproject.toml` as:
```
[project.entry-points."napari.manifest"]
aydin = "aydin.napari_plugin:napari.yaml"
```

Installing Aydin automatically registers the plugin — no extra steps needed.

## Components

| Component | Access | Description |
|-----------|--------|-------------|
| Denoising Widget | Plugins > Aydin > Aydin Denoising Widget | Dock widget for selecting method, layer, and dimensions |
| Aydin Studio | Plugins > Aydin > Aydin Studio | Launches full Aydin Studio with napari layers pre-loaded |
| Context actions | Right-click image layer | "Denoise (high quality)" (FGR-CB) and "Denoise (fast)" (Butterworth) |
| Sample data | File > Open Sample > Aydin | 2 synthetic + 16 Zenodo-hosted microscopy images |

### Denoising Widget (`_widget.py`)

The main widget provides:
- Image layer selection (auto-refreshes when layers change)
- 8 denoising methods (FGR-CatBoost, FGR-LightGBM, CNN-UNet, Butterworth, Gaussian, Spectral, GM, Auto)
- Automatic axis detection with batch/channel override dropdowns
- Background worker thread with progress bar and cancel support

### Studio Bridge (`_studio_bridge.py`)

Opens Aydin Studio as a separate `QMainWindow` without creating a new `QApplication`:
- Selected napari layers (or all if none selected) are copied into Studio's `DataModel`
- Dimensions tab pre-populated from napari axis metadata
- Denoised results pushed back to the same napari viewer

### Axes Utilities (`_axes_utils.py`)

Maps napari axis labels to Aydin's `FileMetadata`:
- Recognises common labels (`t`, `z`, `y`, `x`, `c` and variants)
- Falls back to shape-based heuristics when labels are generic
- Detects batch/channel axes for Aydin's train/denoise API

## Related Packages

- [`../gui/`](../gui/README.md) — Aydin Studio GUI (launched by the Studio bridge)
- [`../restoration/denoise/`](../restoration/denoise/README.md) — Denoiser classes used by both widget and context actions
- [`../io/`](../io/README.md) — `FileMetadata` dataclass and example dataset downloads
