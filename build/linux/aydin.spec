# -*- mode: python ; coding: utf-8 -*-
import sys
print("python ver:", sys.version[:3])
# if sys.version[:3] < "3.7":
#     sys.exit()
# else:
#     print("python version okay")

from glob import glob

def get_qt5_binaries():

    qt_sos = glob("/home/royerlab/anaconda3/envs/aydin/lib/python3.6/site-packages/PyQt5/*.so")

    return [(so,os.path.basename(so)) for so in qt_sos]

import vispy.glsl
import vispy.io
import distributed
import dask
import napari
import gdown
# import m2cgen

from distutils.sysconfig import get_python_lib

from os import path
skimage_plugins = Tree(
    path.join(get_python_lib(), "skimage","io","_plugins"),
    prefix=path.join("skimage","io","plugins"),
)

block_cipher = None

binaries = get_qt5_binaries()

a = Analysis(['../../aydin/cli/cli.py'],
             # pathex=['/Users/ahmetcan.solak/Dev/AhmetCanSolak/aydin'],
             binaries=binaries,
             datas=[

             ('/home/acs-ws/Dev/acs/aydin-ahmetcan/aydin/gui/resources/', '.'),

                    (os.path.join(os.path.dirname(napari.__file__)), 'napari'),
                    (os.path.join(os.path.dirname(dask.__file__)), 'dask'),
                    (os.path.join(os.path.dirname(distributed.__file__)), 'distributed'),
                    (os.path.dirname(vispy.glsl.__file__), os.path.join("vispy", "glsl")),
                    (os.path.join(os.path.dirname(vispy.io.__file__), "_data"), os.path.join("vispy", "io", "_data"))],
             hiddenimports=["aydin.it.transforms.attenuation",
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
                            "numba.core.typing.cffi_utils", "gdown.download", "vispy.app.backends._pyqt5","vispy.glsl"
                                                                     "sklearn.utils._cython_blas"],
             hookspath=["hooks"],
             runtime_hooks=[
                "runtimehooks/hook-bundle.py",
                "runtimehooks/hook-multiprocessing.py",
                "runtimehooks/hook-splash.py"
             ],
             excludes=[])



# filter binaries.. exclude some dylibs that pyinstaller packaged but
# we actually dont need (e.g. wxPython)

import re
reg = re.compile(".*(grpc|PyQt5\.Qt|mpl-data|tcl|zmq|QtWebKit|wxPython|docutils|alabaster|sqlite|plotly|sphinx|msgpack).*")
a.binaries = [s for s in a.binaries if reg.match(s[1]) is None]

a.datas += [("biohub_logo.png", "/home/merlin-workstation/Desktop/aydin test/Dev/aydin-ahmetcan/aydin/gui/resources/biohub_logo.png", 'DATA')]

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
          console=False)

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

