from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

datas = copy_metadata('gdown')
datas += collect_data_files("gdown")
hiddenimports = collect_submodules('gdown')
