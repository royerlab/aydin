from __future__ import print_function


import sys
import os

from build.osx.hooks.hook_utils import _my_collect_data_files

sys.path.append(os.path.split(os.path.abspath(__file__))[0])


datas = _my_collect_data_files("aydin", flatten_dirs=True)


print("\n"*5)
print(datas)
print("\n"*5)
