import numpy
import pytest


@pytest.fixture(autouse=True)
def run_around_tests():
    # A test function will be run at this point
    yield


@pytest.fixture(scope="session")
def sample_2d_image():
    """Normalized 2D test image."""
    from skimage.data import camera

    from aydin.io.datasets import normalise

    return normalise(camera().astype(numpy.float32))


@pytest.fixture(scope="session")
def sample_2d_noisy_pair():
    """Clean/noisy image pair for denoising tests."""
    from skimage.data import camera

    from aydin.io.datasets import add_noise, normalise

    clean = normalise(camera().astype(numpy.float32))
    noisy = add_noise(clean, seed=42)
    return clean, noisy


def pytest_addoption(parser):
    parser.addoption(
        "--runheavy", action="store_true", default=False, help="run heavy tests"
    )
    parser.addoption(
        "--rungpu", action="store_true", default=False, help="run gpu tests"
    )
    parser.addoption(
        "--rununstable", action="store_true", default=False, help="run unstable tests"
    )


def pytest_collection_modifyitems(config, items):
    # Heavy tests: exclusive behavior
    # Without --runheavy: skip heavy tests
    # With --runheavy: run ONLY heavy tests
    if not config.getoption("--runheavy"):
        skip_heavy = pytest.mark.skip(reason="need --runheavy option to run")
        for item in items:
            if "heavy" in item.keywords:
                item.add_marker(skip_heavy)
    else:
        skip_not_heavy = pytest.mark.skip(
            reason="running only heavy tests (--runheavy)"
        )
        for item in items:
            if "heavy" not in item.keywords:
                item.add_marker(skip_not_heavy)

    # GPU tests: exclusive behavior
    # Without --rungpu: skip gpu tests
    # With --rungpu: run ONLY gpu tests
    if not config.getoption("--rungpu"):
        skip_gpu = pytest.mark.skip(reason="need --rungpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    else:
        skip_not_gpu = pytest.mark.skip(reason="running only gpu tests (--rungpu)")
        for item in items:
            if "gpu" not in item.keywords:
                item.add_marker(skip_not_gpu)

    # Unstable tests: exclusive behavior
    # Without --rununstable: skip unstable tests
    # With --rununstable: run ONLY unstable tests
    if not config.getoption("--rununstable"):
        skip_unstable = pytest.mark.skip(reason="need --rununstable option to run")
        for item in items:
            if "unstable" in item.keywords:
                item.add_marker(skip_unstable)
    else:
        skip_not_unstable = pytest.mark.skip(
            reason="running only unstable tests (--rununstable)"
        )
        for item in items:
            if "unstable" not in item.keywords:
                item.add_marker(skip_not_unstable)
