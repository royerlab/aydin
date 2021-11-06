import sys
import os

sys.path.append(os.path.split(os.path.abspath(__file__))[0])
# from hook_utils import _my_collect_data_files

# datas = _my_collect_data_files("pyopencl", include_py_files = True)

from PyInstaller.utils.hooks import copy_metadata, collect_data_files

datas = copy_metadata('pyopencl')
datas += collect_data_files('pyopencl')


print("\n" * 5)
print(datas)
print("\n" * 5)
