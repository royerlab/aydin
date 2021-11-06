import sys
import os
# sys.path.append(os.path.split(os.path.abspath(__file__))[0])
# from hook_utils import _my_collect_data_files
#
# datas = _my_collect_data_files("sklearn", include_py_files = True)

from PyInstaller.utils.hooks import collect_data_files
datas = collect_data_files('sklearn')
#
# print("\n"*5)
# print(datas)
# print("\n"*5)

from PyInstaller.utils.hooks import is_module_satisfies

if is_module_satisfies("scikit_learn >= 0.23"):
    hiddenimports = ['threadpoolctl', ]

hiddenimports = ['sklearn.utils._cython_blas', 'sklearn.tree._utils', 'sklearn.utils._weight_vector']

