# PyInstaller hook for numba (Linux)
# This hook ensures numba's jitclass module and its _box extension are collected

import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    get_module_file_attribute,
)

# Collect all numba submodules including experimental
hiddenimports = collect_submodules('numba')
hiddenimports += collect_submodules('numba.experimental')
hiddenimports += collect_submodules('numba.experimental.jitclass')

# Explicitly add the jitclass modules
hiddenimports += [
    'numba.experimental',
    'numba.experimental.jitclass',
    'numba.experimental.jitclass.base',
    'numba.experimental.jitclass.boxing',
    'numba.experimental.jitclass.decorators',
    'numba.experimental.jitclass.overloads',
]

# Collect data files
datas = collect_data_files('numba')

# Collect dynamic libraries (including _box.so)
binaries = collect_dynamic_libs('numba')

# Explicitly add the _box binary
try:
    numba_path = os.path.dirname(get_module_file_attribute('numba'))
    jitclass_path = os.path.join(numba_path, 'experimental', 'jitclass')
    if os.path.exists(jitclass_path):
        for f in os.listdir(jitclass_path):
            if f.startswith('_box') and f.endswith('.so'):
                binaries.append(
                    (
                        os.path.join(jitclass_path, f),
                        os.path.join('numba', 'experimental', 'jitclass'),
                    )
                )
except Exception:
    pass
