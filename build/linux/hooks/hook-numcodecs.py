# PyInstaller hook for numcodecs (Linux)
import os
import sys

# Add common hooks to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
from hook_utils import _my_collect_data_files

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

datas = copy_metadata('numcodecs')
datas += collect_data_files("numcodecs")
datas += _my_collect_data_files("numcodecs", include_py_files=True)
hiddenimports = collect_submodules('numcodecs')
