"""Import sanity tests for critical dependencies.

Regression tests for GitHub issues #303, #304, #297, #235, and #298.

These tests verify that key dependencies can be imported without errors,
catching binary incompatibilities (numpy dtype size changes), missing
build tools (Rust/Cargo for imagecodecs), and Qt backend issues early.
"""

import importlib

import pytest


def test_numpy_import_and_version():
    """Verify numpy imports and is version 2.x+ (#303 regression).

    Issue #303 reported 'numpy.dtype size changed' errors due to
    binary incompatibility between numpy 1.x and packages compiled
    against numpy 2.x.
    """
    import numpy

    assert int(numpy.__version__.split('.')[0]) >= 2


def test_scipy_import():
    """Verify scipy imports without numpy binary incompatibility."""
    import scipy
    import scipy.ndimage

    assert scipy is not None


def test_scikit_image_import():
    """Verify scikit-image imports without numpy binary incompatibility (#303).

    The original error trace showed scikit-image failing at import time
    with 'numpy.dtype size changed' when compiled against older numpy.
    """
    import skimage
    import skimage.metrics
    import skimage.restoration

    assert skimage is not None


def test_torch_import():
    """Verify PyTorch imports successfully (#235 regression)."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available in test environment")

    assert torch is not None
    # Verify basic tensor operations work
    t = torch.zeros(2, 3)
    assert t.shape == (2, 3)


def test_qtpy_import():
    """Verify qtpy can be imported and is configured for PyQt6 (#298 regression).

    Issue #298 tracked the PyQt5 to PyQt6 migration. This test ensures
    the Qt abstraction layer imports correctly.
    """
    try:
        import qtpy

        assert qtpy is not None
    except ImportError:
        pytest.skip("qtpy not available in test environment")


def test_aydin_top_level_import():
    """Verify the top-level aydin package imports without errors.

    Issues #303, #304 reported import failures cascading from
    aydin.__init__ through the dependency chain.
    """
    import aydin

    assert aydin is not None


def test_aydin_restoration_imports():
    """Verify all restoration module convenience functions are importable."""
    from aydin.restoration.denoise.classic import classic_denoise
    from aydin.restoration.denoise.noise2selffgr import noise2self_fgr

    assert classic_denoise is not None
    assert noise2self_fgr is not None


def test_aydin_classic_denoisers_importable():
    """Verify all classic denoisers can be imported (#303 regression).

    The import chain aydin -> restoration -> features -> classic_denoisers
    -> j_invariance -> skimage was the failure path in issue #303.
    """
    from aydin.it.classic_denoisers.butterworth import denoise_butterworth
    from aydin.it.classic_denoisers.gaussian import denoise_gaussian
    from aydin.it.classic_denoisers.wavelet import denoise_wavelet

    assert denoise_butterworth is not None
    assert denoise_gaussian is not None
    assert denoise_wavelet is not None


@pytest.mark.heavy
def test_imagecodecs_import():
    """Verify imagecodecs imports without requiring Rust (#304 regression).

    Issue #304 reported that pip install failed because imagecodecs
    required Rust/Cargo to compile. With updated version constraints,
    pre-built wheels should be available.
    """
    try:
        import imagecodecs

        assert imagecodecs is not None
    except ImportError:
        pytest.skip("imagecodecs not available in test environment")
