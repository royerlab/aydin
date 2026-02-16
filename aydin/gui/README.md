# GUI — Aydin Studio (`aydin/gui/`)

This package implements the PyQt6-based graphical user interface (Aydin Studio) for interactive image denoising.

## Architecture

```
aydin/gui/
├── gui.py              # App (QMainWindow) — main application window
├── main_page.py        # MainPage (QWidget) — central tabbed workflow widget
├── tabs/               # Workflow tabs and shared DataModel
│   ├── data_model.py   # DataModel — shared state for loaded images
│   └── qt/             # PyQt6 tab implementations
├── _qt/                # Low-level Qt utilities and widgets
│   ├── custom_widgets/ # Reusable custom widgets
│   └── job_runners/    # Background denoising/preview workers
└── resources/          # JSON resources and configuration
```

### Application Entry Point

`gui.py` defines:
- `App(QMainWindow)` — Main application window with menu bar and status
- `run(ver)` — Entry point that initializes the Qt application and shows the window
- Linux Qt6 dependency checking

### Workflow

Aydin Studio presents a tabbed workflow:

```
Summary | File(s) | Image(s) | Dimensions | Training Crop | Denoising Crop | Pre/Post-Processing | Denoise
```

Each tab is a PyQt6 widget in [`tabs/qt/`](tabs/README.md). The `MainPage` widget manages tab navigation, coordinates denoising jobs, and displays results via an integrated napari viewer. The Denoising Crop tab can be disabled when using the same crop region as training.

### Shared State

`DataModel` (`tabs/data_model.py`) is the central state object shared across all tabs:
- Holds loaded images as `ImageRecord` dataclass instances
- Tracks file paths, output folders, dimension assignments, and denoising flags
- Manages transform and denoiser configuration

## Subpackages

| Subpackage | Purpose | Details |
|------------|---------|---------|
| [`tabs/`](tabs/README.md) | Tab implementations + DataModel | Workflow tabs and shared state |
| [`_qt/`](_qt/README.md) | Qt utilities and widgets | Custom widgets, job runners, transform UI |
| `resources/` | JSON resource loading | Configuration data and defaults |

## Important: HTML Docstrings

Tab class docstrings in `tabs/qt/` use custom HTML-like tags rendered by the GUI:
- **`<moreless>`** — Creates expandable "Read more / Read less" sections
- **`<split>`** — Creates two-column layout within `<moreless>` sections
- **`<br>`** — HTML line breaks

These tags are parsed by `QReadMoreLessLabel` in `_qt/custom_widgets/readmoreless_label.py`. **Do not remove or convert these tags.**

See [CLAUDE.md](../../CLAUDE.md) for full docstring conventions.

## For Contributors

- **Add a new workflow tab**: Create a QWidget in `tabs/qt/`, add it to the tab sequence in `main_page.py`
- **Add a new custom widget**: Create it in `_qt/custom_widgets/`
- **Add a background job**: Create a job runner in `_qt/job_runners/`

## Related Packages

- [`../restoration/denoise/`](../restoration/denoise/README.md) — Denoiser classes whose arguments drive the Denoise tab UI
- [`../it/transforms/`](../it/transforms/README.md) — Transform classes displayed in the Processing tab
- [`../cli/`](../cli/README.md) — CLI alternative to the GUI; `aydin` with no args launches the GUI
- [`../io/`](../io/README.md) — Image I/O for loading and saving files
