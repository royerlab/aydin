"""Tests for the variance stabilisation transform (VST)."""

import numpy
from skimage.data import binary_blobs

from aydin.analysis.camera_simulation import simulate_camera_image
from aydin.io.datasets import characters
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform


def test_vst():
    """Test VST preprocess/postprocess round-trip for all supported modes."""
    image = characters()
    image = image.astype(numpy.float32) * 0.1
    noisy = simulate_camera_image(image)

    for mode in [
        'box-cox',
        'yeo-johnson',
        'quantile',
        'anscomb',
        'log',
        'sqrt',
        'identity',
    ]:
        print(f"testing mode: {mode}")
        vst = VarianceStabilisationTransform(mode=mode, leave_as_float=False)

        preprocessed = vst.preprocess(noisy)
        postprocessed = vst.postprocess(preprocessed)

        error = numpy.abs(
            postprocessed.astype(numpy.float32) - noisy.astype(numpy.float32)
        ).mean()

        print(f"round-trip error: {error}")

        assert error < 0.33

        assert postprocessed.dtype == noisy.dtype
        assert postprocessed.shape == noisy.shape


def test_vst_3d():
    """Test variance stabilization transform on 3D volume."""
    image = binary_blobs(length=32, n_dim=3, rng=1).astype(numpy.float32) * 0.1
    noisy = simulate_camera_image(image)

    # Test a subset of modes that are commonly used
    for mode in ['anscomb', 'log', 'sqrt', 'identity']:
        print(f"testing 3D mode: {mode}")
        vst = VarianceStabilisationTransform(mode=mode, leave_as_float=False)

        preprocessed = vst.preprocess(noisy)
        postprocessed = vst.postprocess(preprocessed)

        error = numpy.abs(
            postprocessed.astype(numpy.float32) - noisy.astype(numpy.float32)
        ).mean()

        print(f"round-trip error: {error}")

        assert error < 0.33
        assert preprocessed.shape == noisy.shape
        assert postprocessed.dtype == noisy.dtype
        assert postprocessed.shape == noisy.shape
