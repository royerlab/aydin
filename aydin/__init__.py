"""Aydin: self-supervised, auto-tuned image denoising for n-dimensional images.

Provides classical, patch-based, and machine-learning denoisers accessible
through a Python API, CLI, or the Aydin Studio GUI.
"""

__version__ = "2025.2.4"

_LAZY_IMPORTS = {
    'Classic': 'aydin.restoration.denoise.classic',
    'classic_denoise': 'aydin.restoration.denoise.classic',
    'noise2self_cnn': 'aydin.restoration.denoise.noise2selfcnn',
    'noise2self_fgr': 'aydin.restoration.denoise.noise2selffgr',
}

__all__ = [
    "noise2self_fgr",
    "noise2self_cnn",
    "Classic",
    "classic_denoise",
    "__version__",
]


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr  # cache for subsequent access
        return attr
    raise AttributeError(f"module 'aydin' has no attribute {name!r}")
