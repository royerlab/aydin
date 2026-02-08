# PyInstaller hook for numba (macOS)
# This hook ensures numba's jitclass module and its _box extension are collected

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# Collect all numba submodules
hiddenimports = collect_submodules('numba')

# Collect data files
datas = collect_data_files('numba')

# Collect dynamic libraries (including _box.dylib)
binaries = collect_dynamic_libs('numba')
