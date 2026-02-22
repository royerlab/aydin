# GUI Tabs (`aydin/gui/tabs/`)

This subpackage contains the workflow tab implementations and shared data model for Aydin Studio.

## Architecture

```
aydin/gui/tabs/
├── data_model.py       # DataModel — shared state across all tabs
└── qt/                 # PyQt6 tab widget implementations
    ├── files.py        # FilesTab — file selection and loading
    ├── dimensions.py   # DimensionsTab — axis assignment (batch/channel/spatial)
    ├── images.py       # ImagesTab — image preview and selection
    ├── base_cropping.py        # BaseCroppingTab — shared cropping UI
    ├── training_cropping.py    # TrainingCroppingTab — crop region for training
    ├── denoising_cropping.py   # DenoisingCroppingTab — crop region for denoising
    ├── denoise.py      # DenoiseTab — algorithm/variant selection, parameters
    ├── processing.py   # ProcessingTab — transform pipeline configuration
    └── summary.py      # SummaryTab — results display, export, citation
```

### Tab Order

```
SummaryTab | FilesTab | ImagesTab | DimensionsTab | TrainingCroppingTab | DenoisingCroppingTab | ProcessingTab | DenoiseTab
```

The DenoisingCroppingTab can be disabled when using the same crop region as training.

### `DataModel` (`data_model.py`)

Central state object shared by all tabs:

- **`ImageRecord`** (dataclass) — Represents a single loaded image with: `filename`, `array`, `metadata`, `denoise` flag, `filepath`, `output_folder`
- **`DataModel`** — Manages the collection of `ImageRecord` instances, file paths, dimension assignments, transform settings, and denoiser configuration

All tabs read from and write to the same `DataModel` instance, ensuring consistent state across the workflow.

## Important: HTML Docstrings

Tab class docstrings use custom HTML-like tags parsed by `QReadMoreLessLabel`:

- **`<moreless>`** — Wraps content that becomes an expandable "Read more / Read less" section
- **`<split>`** — Divides content within `<moreless>` into a two-column layout
- **`<br>`** — Standard HTML line breaks

These tags appear in: `denoise.py`, `processing.py`, `dimensions.py`, `summary.py`, `base_cropping.py`

**Do not remove or convert these tags.** See [CLAUDE.md](../../../CLAUDE.md) for full docstring conventions.

## For Contributors

To add a new workflow tab:

1. Create a QWidget subclass in `qt/`
2. Read from / write to `DataModel` for shared state
3. Add the tab to the tab sequence in `../main_page.py`
4. If the tab has a descriptive docstring, use `<moreless>` and `<split>` for expandable content

## Related Packages

- [`../`](../README.md) — Parent GUI package (main window and workflow coordination)
- [`../_qt/`](../_qt/README.md) — Custom widgets and job runners used by tabs
- [`../../restoration/denoise/`](../../restoration/denoise/README.md) — Denoiser argument metadata drives the Denoise tab
