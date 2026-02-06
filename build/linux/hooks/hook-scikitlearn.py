# PyInstaller hook for scikit-learn (Linux)
from PyInstaller.utils.hooks import collect_data_files, is_module_satisfies

datas = collect_data_files('sklearn')

hiddenimports = [
    'sklearn.utils._cython_blas',
    'sklearn.tree._utils',
    'sklearn.utils._weight_vector',
]

if is_module_satisfies("scikit_learn >= 0.23"):
    hiddenimports.append('threadpoolctl')
