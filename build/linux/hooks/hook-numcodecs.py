import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata
from build.osx.hooks.hook_utils import _my_collect_data_files

sys.path.append(os.path.split(os.path.abspath(__file__))[0])

datas = copy_metadata('numcodecs')
datas += collect_data_files("numcodecs")
datas += _my_collect_data_files("numcodecs", include_py_files=True)
hiddenimports = collect_submodules('numcodecs')
