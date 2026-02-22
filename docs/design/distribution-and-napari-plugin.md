# Aydin Distribution & Napari Plugin Design

## Overview

This document captures the design discussion for moving Aydin away from PyInstaller-based bundling toward a modern, multi-tier distribution strategy.

---

## Three-Tier Distribution Strategy

### Tier 1: Napari Plugin (Simple Widget)

A dock widget inside napari for quick denoising of the currently selected layer.

```
+---------------------------+
|  Aydin Denoiser           |
|                           |
|  Layer: [dropdown v]      |
|  Method: [Butterworth v]  |
|           GM              |
|           Spectral        |
|           N2S-FGR-cb      |
|                           |
|  [Denoise]  [Preview]     |
|                           |
|  ------------------------ |
|  [Open Aydin Studio...]   |
+---------------------------+
```

- Denoises the selected napari layer in-place (result appears as a new layer)
- Only exposes non-advanced denoisers (Butterworth, GM, Spectral, N2S-FGR-cb/lgbm)
- Calls directly into `aydin.restoration.denoise`, bypassing the Aydin GUI layer
- Minimal code (~200 lines), uses `magicgui` or raw Qt `QWidget`
- Published on napari hub for discoverability

### Tier 2: Full Aydin Studio (Separate Window from napari)

Launched from the plugin widget via "Open Aydin Studio..." button. Provides the complete workflow: file management, dimension assignment, training/denoising crops, transforms, all denoisers (including advanced mode), pretrained models, etc.

- Implemented as a `QMainWindow` with `parent=viewer.window._qt_window`
- Reuses the existing `MainPage` widget and full GUI code as-is
- Keeps Aydin's standalone identity and full feature set

### Tier 3: Docker

- **CLI image**: For batch processing and HPC use. Slim image, GPU support via `--gpus all`. Compatible with Singularity/Apptainer for HPC clusters.
- **GUI image** (optional): Xpra-based for browser access (`http://localhost:PORT`)

---

## Napari-Aydin Data Flow Design

### Current Data Flow (Standalone Aydin Studio)

```
Files on disk -> FilesTab -> DataModel -> ImagesTab -> DimensionsTab -> CropTabs -> Denoise
                                                                                      |
                                                                              napari.Viewer()  <- NEW window
```

Everything starts from file paths. The `DataModel` holds file paths, reads them into arrays, and passes them through the pipeline. At the end, "View images" creates a brand new napari viewer.

### New Data Flow (Launched from napari)

#### Scenario A: User selects layers in napari, then opens Aydin Studio

```
napari viewer (layers already loaded)
       |
  "Open Aydin Studio..."
       |
  Aydin Studio receives layer data as numpy arrays
       |
  Skip FilesTab/ImagesTab (data already loaded)
       |
  DimensionsTab -> CropTabs -> Denoise
       |
  Results pushed back as new layers in the SAME napari viewer
```

#### Scenario B: User opens Aydin Studio and loads files from within it

```
napari viewer
       |
  "Open Aydin Studio..."
       |
  User loads files via FilesTab (traditional workflow)
       |
  Full pipeline as usual
       |
  Results pushed to the SAME napari viewer
```

### Key Design Decisions

#### 1. How do images get from napari to Aydin?

When launching Aydin Studio from the napari plugin, pass a reference to the `napari.Viewer`. Aydin Studio can then:

- Read selected layers: `viewer.layers.selection` gives the numpy arrays directly
- Pre-populate the DataModel with those arrays instead of file paths
- Skip directly to Dimensions tab (or auto-detect dimensions from napari's axis metadata)

This requires extending `DataModel` -- currently it only accepts file paths via `add_filepaths()`. It would need an `add_arrays()` method that takes numpy arrays + metadata directly.

#### 2. What does "View images" do?

**It reuses the existing napari viewer, not spawning a new one.** Currently `MainPage.open_images_with_napari()` does `viewer = napari.Viewer()` -- a new window every time. When launched from napari, it should instead add layers to the viewer that's already open:

```python
# Current (standalone):
viewer = napari.Viewer()
viewer.add_image(result, name="denoised")

# New (launched from napari):
self.napari_viewer.add_image(result, name="denoised")  # reuse existing
```

The button text changes from "View images" to "Send to napari" to make the action clearer.

#### 3. Bidirectional data flow

- **napari -> Aydin Studio**: Selected layers become input images
- **Aydin Studio -> napari**: Denoised results appear as new layers
- User can compare noisy vs denoised side-by-side in napari (grid view, opacity sliders, etc.)
- If the user wants to re-denoise with different settings, the layers are already there

#### 4. FilesTab behavior

Keep FilesTab functional regardless of launch mode:

- If images came from napari layers, the FilesTab shows them as "napari: layer_name" instead of file paths
- Users can also add files from disk as usual
- This keeps Aydin Studio fully functional regardless of how it was launched

#### 5. Dimension metadata auto-population

napari layers carry axis labels (e.g., `['t', 'z', 'y', 'x']`). When images come from napari, DimensionsTab can be pre-populated automatically -- potentially skipping that step entirely for the user.

---

## Required Code Changes

| Component | Change |
|-----------|--------|
| `DataModel` | Add `add_arrays(arrays, names, metadata)` alongside `add_filepaths()` |
| `MainPage.__init__` | Accept optional `napari_viewer` parameter |
| `MainPage.open_images_with_napari` | Reuse existing viewer when available, add layers to it |
| `FilesTab` | Display "napari: layer_name" for array-sourced images |
| `DimensionsTab` | Auto-populate from napari axis metadata when available |
| Napari plugin widget | Pass `viewer` reference when launching Aydin Studio |
| "View images" button | Rename to "Send to napari" when launched from napari |

The core denoising engine, transforms, regression, features -- none of that changes. This is purely a data flow / GUI integration task.

---

## New Package Structure

- **Package name**: `aydin-napari` (separate package) or add napari plugin metadata to `aydin` itself
- **`napari.yaml` manifest**: Register the simple widget + optionally a reader for CZI/ND2
- **Publish to**: PyPI and napari hub

---

## Docker Distribution Details

### CLI Image

```bash
# Basic usage
docker run --rm -v /data:/data aydin denoise /data/image.tif

# With GPU
docker run --rm --gpus all -v /data:/data aydin denoise /data/image.tif
```

### GUI Image (optional, via Xpra)

```bash
docker run --rm -p 9876:9876 aydin-studio
# Open http://localhost:9876 in browser
```

### HPC Compatibility

```bash
# Convert Docker image to Singularity/Apptainer
singularity pull docker://aydin:latest
singularity run --nv aydin_latest.sif denoise image.tif
```

---

## Architectural Notes

Aydin's core is already well-separated, making this strategy feasible without major refactoring:

```
aydin.restoration.denoise  <- algorithms (standalone, no GUI dependency)
aydin.it.*                 <- image translator framework
aydin.gui.*                <- Qt widgets (can be launched independently)
aydin.cli.*                <- CLI (perfect for Docker)
```

The napari plugin calls into `aydin.restoration.denoise` directly. Full Aydin Studio reuses the existing GUI code. Docker wraps the CLI. No code duplication needed.

---

## Alternatives Considered

- **conda-constructor**: Produces native OS installers (.exe/.pkg/.sh). Used by napari, Spyder, MNE-Python. Good for standalone distribution but not pursued as primary approach since the napari plugin strategy eliminates most bundling needs.
- **PyInstaller**: Was the previous approach. Removed due to brittleness and complex spec files.
- **Nuitka**: Python-to-C compiler. Does not support PyQt6 (recommends PySide6). Ruled out.
- **PyOxidizer**: Abandoned (dead since Jan 2023). Ruled out.
- **Gradio/Hugging Face Spaces**: Could complement the strategy as a zero-install web demo for small images.
- **pixi + pixi-pack**: Modern conda alternative. Good for dev workflow but pixi-pack produces .tar archives, not native installers. Could be used for development while conda-constructor handles end-user installers if standalone installers are needed in the future.
