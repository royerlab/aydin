# Aydin PyInstaller Builds

This directory contains the PyInstaller configuration and build scripts for creating standalone Aydin distributions for different platforms.

## Prerequisites

- **Python 3.9+** (3.9, 3.10, 3.11, or 3.12)
- **PyInstaller 5.x+**
- **Aydin installed in development mode**: `pip install -e ".[bundle]"`

The bundle dependencies include pinned versions for reproducibility during PyInstaller builds.

## Building

### Linux

```bash
cd build/linux
./build.sh
```

Output: `dist/aydin/` directory with the `aydin` executable

### Windows

```cmd
cd build\win
build.bat
```

Output: `dist\aydin\` directory with `aydin.exe`

### macOS

```bash
cd build/osx
./build.sh
```

Output:
- `dist/Aydin.app` - macOS application bundle
- `dist/aydin/` - Directory distribution

## Running the Built Application

### Linux
```bash
cd dist
./run_aydin.sh
# or directly:
./aydin/aydin
```

### Windows
```cmd
cd dist
run_aydin.bat
:: or directly:
aydin\aydin.exe
```

### macOS
```bash
open dist/Aydin.app
# or directly:
./dist/aydin/aydin
```

## Directory Structure

```
build/
├── README.md              # This file
├── common/                # Shared hook utilities
│   └── hook_utils.py      # Common functions for PyInstaller hooks
├── linux/
│   ├── aydin.spec         # PyInstaller spec file
│   ├── build.sh           # Build script
│   ├── run_aydin.sh       # Launcher script (copied to dist/)
│   ├── hooks/             # PyInstaller hooks
│   │   ├── hook_utils.py  # Re-exports from common
│   │   ├── hook-lightgbm.py
│   │   ├── hook-scipy.py
│   │   ├── hook-scikitlearn.py
│   │   └── ...
│   └── runtimehooks/      # Runtime hooks
│       ├── hook-bundle.py
│       └── hook-multiprocessing.py
├── osx/
│   ├── aydin.spec         # PyInstaller spec file
│   ├── build.sh           # Build script
│   ├── Info.plist         # macOS app bundle info
│   ├── icon-windowed.icns # macOS app icon
│   ├── hooks/             # PyInstaller hooks
│   └── runtimehooks/      # Runtime hooks
└── win/
    ├── aydin.spec         # PyInstaller spec file
    ├── build.bat          # Build script
    ├── run_aydin.bat      # Launcher script (copied to dist/)
    ├── hooks/             # PyInstaller hooks
    └── runtimehooks/      # Runtime hooks
```

## Troubleshooting

### Common Issues

#### 1. enum34 conflicts
The build scripts automatically uninstall enum34 which can cause issues with Python 3.x. If you see enum-related errors, run:
```bash
pip uninstall -y enum34
```

#### 2. imagecodecs conflicts
The build scripts automatically uninstall imagecodecs which has complex binary dependencies that don't bundle well. If you need imagecodecs functionality, the bundled app won't support all image formats.

#### 3. Missing modules at runtime
If you see `ModuleNotFoundError` at runtime, add the missing module to the `hiddenimports` list in the spec file.

#### 4. Missing data files
If resources or data files are missing, add them to the `datas` list in the spec file.

#### 5. DLL/shared library issues (Windows/Linux)
If you see errors about missing DLLs or .so files, you may need to add a custom hook in the `hooks/` directory. See existing hooks for examples.

### Adding Hidden Imports

Edit the relevant spec file and add modules to the `hiddenimports` list:

```python
hiddenimports=[
    # ... existing imports ...
    "your.missing.module",
]
```

### Adding Data Files

Edit the relevant spec file and add to the `datas` list:

```python
datas=[
    # ... existing data ...
    (source_path, destination_path),
]
```

### Creating Custom Hooks

Create a new file in `hooks/` named `hook-<package>.py`:

```python
# hook-mypackage.py
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files('mypackage')
hiddenimports = collect_submodules('mypackage')
```

## Runtime Hooks

Runtime hooks run at application startup:

- **hook-bundle.py**: Sets `BUNDLED_AYDIN=1` environment variable to signal the app is running as a bundle
- **hook-multiprocessing.py**: Sets `JOBLIB_MULTIPROCESSING=0` to avoid multiprocessing issues in frozen applications

## Notes

- The spec files use dynamic path resolution via Python imports, so they work across different environments without hardcoded paths
- Binary filtering is applied to reduce bundle size by excluding unnecessary Qt modules and other large libraries
- The macOS build creates both a `.app` bundle and a directory distribution
