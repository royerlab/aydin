from aydin.restoration.denoise.noise2selffgr import noise2self_fgr  # noqa: F401
from aydin.restoration.denoise.noise2selfcnn import noise2self_cnn  # noqa: F401
from aydin.restoration.denoise.classic import Classic  # noqa: F401
from aydin.restoration.denoise.classic import classic_denoise  # noqa: F401
from aydin.restoration.deconvolve.lr import lucyrichardson  # noqa: F401

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"
