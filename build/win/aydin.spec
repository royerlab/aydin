# -*- mode: python ; coding: utf-8 -*-

import vispy.glsl
import vispy.io
import distributed
import dask
import napari
import gdown
from distutils.sysconfig import get_python_lib

from os import path
skimage_plugins = Tree(
    path.join(get_python_lib(), "skimage","io","_plugins"),
    prefix=path.join("skimage","io","plugins"),
)

block_cipher = None


a = Analysis(['../../aydin/cli/cli.py'],
             # pathex=['/Users/ahmetcan.solak/Dev/AhmetCanSolak/aydin'],
             #binaries = [],
             binaries=[],
             datas=[(r'C:\Users\Royerlab\Dev\acs\aydin-ahmetcan\aydin\gui\resources', r'.'),
                    (os.path.join(os.path.dirname(napari.__file__)), 'napari'),
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
                             "sklearn.neighbors._partition_nodes",
                            "numba.core.typing.cffi_utils",
                            "sklearn.neighbors._partition_nodes",
                            "pydantic",
                             "magicgui",
                             "napari_plugin_engine",
                             "qtpy",
                             "napari._qt",
                             "sklearn.utils._typedefs",
             "gdown.download", "napari", "tensorflow_core._api.v2.compat","vispy.app.backends._pyqt5","vispy.ext._bundled.six",
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
reg = re.compile(".*(PyQt4|mpl-data|tcl|zmq|QtWebKit|QtQuick|wxPython).*")

a.binaries = [s for s in a.binaries if reg.match(s[1]) is None]

a.datas += [("biohub_logo.png", "/Users/ahmetcan.solak/Dev/AhmetCanSolak/aydin/aydin/gui/resources/biohub_logo.png", 'DATA')]

pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=1,
          name='aydin',
          #debug=False,
          bootloader_ignore_signals=False,
          strip=None,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True)

app = BUNDLE(exe,
             a.binaries,
             a.zipfiles,
             name='aydin.app',
             icon=None)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               skimage_plugins,
               strip=False,
               upx=True,
               name='aydin')
