import pytest


@pytest.fixture(autouse=True)
def run_around_tests():
    # Code that will run before your test, for example:
    import os

    # os.environ["PYOPENCL_CTX"] = "0:2"

    # A test function will be run at this point
    yield
    # Code that will run after your test, for example:


def pytest_addoption(parser):
    parser.addoption(
        "--runheavy", action="store_true", default=False, help="run heavy tests"
    )
    parser.addoption(
        "--rungpu", action="store_true", default=False, help="run gpu tests"
    )
    parser.addoption(
        "--runopencl", action="store_true", default=False, help="run opencl tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "heavy: mark test as heavy to run")
    config.addinivalue_line("markers", "gpu: mark test as gpu to run")
    config.addinivalue_line("markers", "opencl: mark test as opencl to run")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runheavy"):
        # --runheavy given in cli: do not skip slow tests
        skip_heavy = pytest.mark.skip(reason="need --runheavy option to run")
        for item in items:
            if "heavy" in item.keywords:
                item.add_marker(skip_heavy)

    if not config.getoption("--rungpu"):
        skip_gpu = pytest.mark.skip(reason="need --rungpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    if not config.getoption("--runopencl"):
        skip_gpu = pytest.mark.skip(reason="need --runopencl option to run")
        for item in items:
            if "opencl" in item.keywords:
                item.add_marker(skip_gpu)
