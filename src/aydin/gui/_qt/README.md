# Qt Utilities (`aydin/gui/_qt/`)

This subpackage contains low-level PyQt6 utilities, custom widgets, and background job runners used throughout Aydin Studio.

## Architecture

```
aydin/gui/_qt/
├── output_wrapper.py           # Stdout/stderr capture for GUI logging
├── qtreewidget_utils.py        # QTreeWidget helper functions
├── transforms_tab_item.py      # Single transform configuration widget
├── transforms_tab_widget.py    # Transform list management widget
├── custom_widgets/             # Reusable custom PyQt6 widgets
└── job_runners/                # Background worker threads
```

### Transform UI

- `transforms_tab_item.py` — Renders a single transform's parameters as a widget. Reads the transform class `__doc__`, applies `strip_notgui()`, and converts `\n` to `<br>` for display.
- `transforms_tab_widget.py` — Manages the ordered list of transforms, enabling add/remove/reorder operations.

## Custom Widgets (`custom_widgets/`)

| Module | Widget | Purpose |
|--------|--------|---------|
| `readmoreless_label.py` | `QReadMoreLessLabel` | Expandable label parsing `<moreless>` and `<split>` tags |
| `range_slider.py` | `QRangeSlider` | Dual-handle range slider control |
| `range_slider_with_labels.py` | `QRangeSliderWithLabels` | Range slider with min/max value labels |
| `activity_widget.py` | `ActivityWidget` | Progress/activity indicator |
| `overlay.py` | `Overlay` | Loading state overlay |
| `constructor_arguments.py` | `ConstructorArgumentsWidget` | Auto-generates UI from constructor parameters |
| `denoise_tab_common.py` | — | Shared denoising UI components |
| `denoise_tab_method.py` | — | Denoising method selector widget |
| `denoise_tab_pretrained_method.py` | — | Pre-trained model selector |
| `program_flow_diagram.py` | — | Visual workflow diagram |
| `system_summary.py` | — | System information display |
| `horizontal_line_break_widget.py` | — | Horizontal separator |
| `vertical_line_break_widget.py` | — | Vertical separator |

## Job Runners (`job_runners/`)

Background job execution using Qt threading:

| Module | Class | Purpose |
|--------|-------|---------|
| `worker.py` | `Worker` / `WorkerSignals` | Generic QRunnable worker thread with signals |
| `denoise_job_runner.py` | `DenoiseJobRunner` | Manages full denoising jobs in background |
| `base_preview_job_runner.py` | `BasePreviewJobRunner` | Base class for preview jobs |
| `preview_job_runner.py` | `PreviewJobRunner` | Single image preview denoising |
| `previewall_job_runner.py` | `PreviewAllJobRunner` | Preview all loaded images |

All job runners use Qt's signal/slot mechanism to communicate progress and results back to the main GUI thread without freezing the UI.

## For Contributors

- **Add a custom widget**: Create it in `custom_widgets/`, subclass `QWidget`
- **Add a background job**: Subclass `BasePreviewJobRunner` or create a new `QRunnable` using `Worker`

## Related Packages

- [`../tabs/`](../tabs/README.md) — Workflow tabs that use these widgets and job runners
- [`../../it/transforms/`](../../it/transforms/README.md) — Transform classes rendered by `transforms_tab_item.py`
