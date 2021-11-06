import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata
from build.osx.hooks.hook_utils import _my_collect_data_files

sys.path.append(os.path.split(os.path.abspath(__file__))[0])

datas = copy_metadata('numba')
datas += collect_data_files("numba")
datas += _my_collect_data_files("numba", include_py_files=False)
hiddenimports = collect_submodules('numba')
