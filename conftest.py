import pytest


@pytest.fixture(autouse=True)
def run_around_tests():
    # Code that will run before your test, for example:
    import os

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
        "--rununstable", action="store_true", default=False, help="run unstable tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "heavy: mark test as heavy to run")
    config.addinivalue_line("markers", "gpu: mark test as gpu to run")
    config.addinivalue_line("markers", "unstable: mark test as unstable to run")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runheavy"):
        # --runheavy given in cli: do not skip slow tests
        skip_heavy = pytest.mark.skip(reason="need --runheavy option to run")
        for item in items:
            if "heavy" in item.keywords:
                item.add_marker(skip_heavy)
    else:
        skip_heavy = pytest.mark.skip(reason="need --runheavy option to run")
        for item in items:
            if "heavy" not in item.keywords:
                item.add_marker(skip_heavy)

    if not config.getoption("--rungpu"):
        skip_gpu = pytest.mark.skip(reason="need --rungpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    if not config.getoption("--rununstable"):
        skip_unstable = pytest.mark.skip(reason="need --rununstable option to run")
        for item in items:
            if "unstable" in item.keywords:
                item.add_marker(skip_unstable)
    else:
        skip_unstable = pytest.mark.skip(reason="need --rununstable option to run")
        for item in items:
            if "unstable" not in item.keywords:
                item.add_marker(skip_unstable)
