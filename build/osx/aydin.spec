# -*- mode: python ; coding: utf-8 -*-
import sys
print("python ver:", sys.version[:3])
if sys.version[:3] < "3.7":
    sys.exit()
else:
    print("python version okay")

import vispy.glsl
import vispy.io
import distributed
import dask
# import m2cgen
import napari
import gdown
from PyInstaller.utils.hooks import get_module_file_attribute

from distutils.sysconfig import get_python_lib

from os import path
skimage_plugins = Tree(
    path.join(get_python_lib(), "skimage","io","_plugins"),
    prefix=path.join("skimage","io","plugins"),
)

block_cipher = None

# (os.path.join(os.path.dirname(get_module_file_attribute('sklearn.cluster')), "_k_means_common.cpython-39-darwin.so"), "sklearn.cluster._k_means_common")

a = Analysis(['../../aydin/cli/cli.py'],
             pathex=['/Users/ahmetcan.solak/Dev/AhmetCanSolak/aydin/aydin/cli'],
             binaries=[],
             datas=[(os.path.join(os.path.dirname(napari.__file__)), 'napari'),
                    (os.path.join(os.path.dirname(dask.__file__)), 'dask'),
                    (os.path.join(os.path.dirname(distributed.__file__)), 'distributed'),
                    (os.path.dirname(vispy.glsl.__file__), os.path.join("vispy", "glsl")),
                    (os.path.join(os.path.dirname(vispy.io.__file__), "_data"), os.path.join("vispy", "io", "_data"))],
             hiddenimports=[
                             "aydin.it.transforms.attenuation",
                             "aydin.it.transforms.deskew",
                             "aydin.it.transforms.fixedpattern",
                             "aydin.it.transforms.highpass",
                             "aydin.it.transforms.histogram",
                             "aydin.it.transforms.motion",
                             "aydin.it.transforms.padding",
                             "aydin.it.transforms.periodic",
                             "aydin.it.transforms.range",
                             "aydin.it.transforms.salt_pepper",
                             "aydin.it.transforms.variance_stabilisation",
                            "numba.core.typing.cffi_utils",
                            "aydin.it.regression.cb",
                            "aydin.it.regression.lgbm",
                            "aydin.it.regression.linear",
                            "aydin.it.regression.nn",
                            "aydin.it.regression.random_forest",
                            "aydin.it.regression.support_vector",
                            "sklearn.neighbors._partition_nodes",
                            "sklearn.cluster.*",
                            "pydantic",
                            "magicgui",
                            "napari_plugin_engine",
                            "qtpy",
                            "imageio.plugins.tifffile",
                             "imageio.plugins.pillow_legacy",
                             "imageio.plugins.ffmpeg",
                             "imageio.plugins.bsdf",
                             "imageio.plugins.dicom",
                             "imageio.plugins.feisem",
                             "imageio.plugins.fits",
                             "imageio.plugins.gdal",
                             "imageio.plugins.simpleitk",
                             "imageio.plugins.npz",
                             "imageio.plugins.spe",
                             "imageio.plugins.swf",
                             "imageio.plugins.grab",
                             "imageio.plugins.lytro",
                             "imageio.plugins.freeimage",
                             "imageio.plugins.freeimagemulti",
                            "napari._qt",
                            "psygnal._signal",
                            "sklearn.utils._typedefs",
             "gdown.download", "napari", "tensorflow_core._api.v2.compat","vispy.app.backends._pyqt5","vispy.glsl",
                                                                     "sklearn.utils._cython_blas"],
             hookspath=["hooks"],
             runtime_hooks=[
                "runtimehooks/hook-bundle.py",
                "runtimehooks/hook-multiprocessing.py",
                "runtimehooks/hook-splash.py"
             ],
             excludes=[])

pyz = PYZ(a.pure)

# filter binaries.. exclude some dylibs that pyinstaller packaged but
# we actually dont need (e.g. wxPython)

import re
reg = re.compile(".*(PyQt4|k_means|mpl-data|zmq|QtWebKit|QtQuick|wxPython).*")

# from pprint import pprint
# pprint(a.binaries)

a.binaries = [s for s in a.binaries if reg.match(s[1]) is None]

a.datas += [("biohub_logo.png", "/Users/ahmetcan.solak/Dev/AhmetCanSolak/aydin/aydin/gui/resources/biohub_logo.png", 'DATA')]

pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=1,
          # a.binaries,
          # a.zipfiles,
          # a.datas,
          name='aydin',
          debug=False,
          #debug=True,
          strip=None,
          upx=True,
          console=True )

app = BUNDLE(exe,
             name='aydin.app',
             upx=True,
             icon=None)


coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='aydin')
