from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

datas = collect_data_files("skimage.io._plugins")
hiddenimports = collect_submodules('skimage.io._plugins')
