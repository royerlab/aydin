# Re-export from common hook_utils for backwards compatibility
# This file allows hooks to import from the local hooks directory

import os
import sys

# Add common directory to path and re-export
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
from hook_utils import (
    _get_toc_objects,
    _module_path,
    _my_collect_data_files,
    get_lightgbm_binary,
    get_platform_binary_extension,
)

__all__ = [
    '_module_path',
    '_get_toc_objects',
    '_my_collect_data_files',
    'get_lightgbm_binary',
    'get_platform_binary_extension',
]
