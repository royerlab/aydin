import os as _os
import sys as _sys

# Suppress verbose TensorFlow/CUDA C++ registration warnings on import
_os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
_stderr_fd = _sys.stderr.fileno()
_saved_stderr = _os.dup(_stderr_fd)
_devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
_os.dup2(_devnull_fd, _stderr_fd)
_os.close(_devnull_fd)
try:
    from aydin.restoration.denoise.classic import Classic  # noqa: F401
    from aydin.restoration.denoise.classic import classic_denoise  # noqa: F401
    from aydin.restoration.denoise.noise2selfcnn import noise2self_cnn  # noqa: F401
    from aydin.restoration.denoise.noise2selffgr import noise2self_fgr  # noqa: F401
finally:
    _os.dup2(_saved_stderr, _stderr_fd)
    _os.close(_saved_stderr)

__version__ = "2025.2.4"

__all__ = [
    "noise2self_fgr",
    "noise2self_cnn",
    "Classic",
    "classic_denoise",
    "__version__",
]
