import sys
import os
from PyInstaller.utils.hooks import get_module_file_attribute, collect_data_files


def _my_collect_data_files(modname, flatten_dirs = False, **kwargs):
    files = collect_data_files(modname, **kwargs)
    if flatten_dirs:
        # files = [(source, os.path.split(dest)[0])for source, dest in files]
        files = [(source, ".") for source, dest in files]

    return files


sys.path.append(os.path.split(os.path.abspath(__file__))[0])
# from hook_utils import _my_collect_data_files

datas = _my_collect_data_files("lightgbm", include_py_files = False)
#
# print("\n"*5)
# print(datas)
# print("\n"*5)
#
import glob
from PyInstaller.compat import is_win


# if we bundle the testing module, this will cause
# `scipy` to be pulled in unintentionally but numpy imports
# numpy.testing at the top level for historical reasons.
# excludedimports = collect_submodules('numpy.testing')

binaries = []

# # #package the DLL bundle that official numpy wheels for Windows ship
# if is_win:
#     dll_glob = os.path.join(os.path.dirname(
#         get_module_file_attribute('lightgbm')), 'extra-dll', "*.dll")
#     if glob.glob(dll_glob):
#         binaries.append((dll_glob, "."))

# binaries.append((os.path.join(os.path.dirname(get_module_file_attribute('lightgbm')), "lib_lightgbm.dll"), "lightgbm"))

binaries.append((os.path.join(os.path.dirname(get_module_file_attribute('lightgbm')), "lib_lightgbm.so"), "lightgbm"))
# binaries.append((os.path.join(os.path.dirname(get_module_file_attribute('lightgbm')), "basic"), "lightgbm"))
# binaries.append((os.path.join(os.path.dirname(get_module_file_attribute('lightgbm')), "callback"), "lightgbm"))
# binaries.append((os.path.join(os.path.dirname(get_module_file_attribute('lightgbm')), "compat"), "lightgbm"))
# binaries.append((os.path.join(os.path.dirname(get_module_file_attribute('lightgbm')), "engine"), "lightgbm"))
# binaries.append((os.path.join(os.path.dirname(get_module_file_attribute('lightgbm')), "libpath"), "lightgbm"))
# binaries.append((os.path.join(os.path.dirname(get_module_file_attribute('lightgbm')), "plotting"), "lightgbm"))
