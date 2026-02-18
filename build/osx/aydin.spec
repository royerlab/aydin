# -*- mode: python ; coding: utf-8 -*-
# Aydin PyInstaller spec file for macOS

import os
import re
import sys
import sysconfig

# Version check
if sys.version_info < (3, 9):
    raise RuntimeError("Python 3.9+ required for building Aydin")

print(f"Building with Python {sys.version}")

# Dynamic path resolution - import modules to get their paths
import dask
import distributed
import napari
import numba
import vispy.glsl
import vispy.io

import aydin

# Get numba path for jitclass
numba_path = os.path.dirname(numba.__file__)
numba_jitclass_path = os.path.join(numba_path, 'experimental', 'jitclass')

# Find numba _box binary (platform-specific)
import glob
numba_box_binaries = []
for box_file in glob.glob(os.path.join(numba_jitclass_path, '_box*.so')) + glob.glob(os.path.join(numba_jitclass_path, '_box*.dylib')):
    numba_box_binaries.append((box_file, os.path.join('numba', 'experimental', 'jitclass')))

# Get site-packages dynamically
site_packages = sysconfig.get_path('purelib')

# Get aydin resources dynamically
aydin_path = os.path.dirname(aydin.__file__)
aydin_resources = os.path.join(aydin_path, 'gui', 'resources')

# Skimage plugins
skimage_plugins = Tree(
    os.path.join(site_packages, "skimage", "io", "_plugins"),
    prefix=os.path.join("skimage", "io", "_plugins"),
)

block_cipher = None

a = Analysis(
    ['../../aydin/cli/cli.py'],
    pathex=[],
    binaries=numba_box_binaries,
    datas=[
        (aydin_resources, 'aydin/gui/resources'),
        (os.path.dirname(napari.__file__), 'napari'),
        (os.path.dirname(dask.__file__), 'dask'),
        (os.path.dirname(distributed.__file__), 'distributed'),
        (os.path.dirname(vispy.glsl.__file__), os.path.join("vispy", "glsl")),
        (os.path.join(os.path.dirname(vispy.io.__file__), "_data"), os.path.join("vispy", "io", "_data")),
        # numba jitclass with _box binary
        (numba_jitclass_path, os.path.join("numba", "experimental", "jitclass")),
    ],
    hiddenimports=[
        # Aydin transforms
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
        # Aydin regression (correct module paths)
        "aydin.regression.cb",
        "aydin.regression.lgbm",
        "aydin.regression.linear",
        "aydin.regression.perceptron",
        "aydin.regression.random_forest",
        "aydin.regression.support_vector",
        # sklearn
        "sklearn.neighbors._partition_nodes",
        "sklearn.utils._typedefs",
        "sklearn.utils._heap",
        "sklearn.utils._sorting",
        "sklearn.utils._vector_sentinel",
        "sklearn.utils._cython_blas",
        # numba
        "numba.core.typing.cffi_utils",
        "numba.experimental.jitclass",
        "numba.experimental.jitclass.boxing",
        "numba.experimental.jitclass._box",
        # GUI/visualization
        "pydantic",
        "magicgui",
        "napari_plugin_engine",
        "qtpy",
        "napari._qt",
        "psygnal._signal",
        "vispy.app.backends._pyqt5",
        "vispy.glsl",
        # imageio plugins
        "imageio.plugins.tifffile",
        "imageio.plugins.pillow",
        "imageio.plugins.ffmpeg",
    ],
    hookspath=["hooks", "../common"],
    runtime_hooks=[
        "runtimehooks/hook-bundle.py",
        "runtimehooks/hook-multiprocessing.py",
    ],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter binaries - exclude unnecessary libraries
exclude_pattern = re.compile(
    r".*(PyQt4|PyQt6|mpl-data|tcl|tk|zmq|QtWebKit|QtQuick|wxPython|"
    r"grpc|docutils|alabaster|sqlite|plotly|sphinx|msgpack).*"
)
a.binaries = [b for b in a.binaries if exclude_pattern.match(b[1]) is None]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='aydin',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    skimage_plugins,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='aydin',
)

# macOS app bundle
app = BUNDLE(
    coll,
    name='Aydin.app',
    icon='icon-windowed.icns',
    bundle_identifier='org.royerlab.aydin',
    info_plist={
        'CFBundleShortVersionString': aydin.__version__,
        'CFBundleVersion': aydin.__version__,
        'NSHighResolutionCapable': 'True',
        'NSPrincipalClass': 'NSApplication',
    },
)
