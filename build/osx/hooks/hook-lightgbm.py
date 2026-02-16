# PyInstaller hook for lightgbm (macOS)
import os
import sys

# Add common hooks to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
from hook_utils import _my_collect_data_files, get_lightgbm_binary

from PyInstaller.utils.hooks import get_module_file_attribute

# Collect data files
datas = _my_collect_data_files("lightgbm", include_py_files=False)

# Collect the platform-specific binary
binaries = []
lgbm_path = os.path.dirname(get_module_file_attribute('lightgbm'))
lib_name = get_lightgbm_binary()
lib_path = os.path.join(lgbm_path, lib_name)

if os.path.exists(lib_path):
    binaries.append((lib_path, "lightgbm"))
