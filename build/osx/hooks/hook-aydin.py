from __future__ import print_function


import sys
import os

from PyInstaller.utils.hooks import collect_data_files


def _my_collect_data_files(modname, flatten_dirs = False, **kwargs):
    files = collect_data_files(modname, **kwargs)
    if flatten_dirs:
        # files = [(source, os.path.split(dest)[0])for source, dest in files]
        files = [(source, ".") for source, dest in files]

    return files


sys.path.append(os.path.split(os.path.abspath(__file__))[0])


datas = _my_collect_data_files("aydin", flatten_dirs=True)


print("\n"*5)
print(datas)
print("\n"*5)
