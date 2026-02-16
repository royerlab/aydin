# PyInstaller hook for aydin (macOS)
import os
import sys

# Add common hooks to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
from hook_utils import _my_collect_data_files

datas = _my_collect_data_files("aydin", flatten_dirs=True)
