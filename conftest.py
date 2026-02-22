import pytest


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
    parser.addoption(
        "--rungui", action="store_true", default=False, help="run gui tests"
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

    # GUI tests: exclusive behavior
    # Without --rungui: skip gui-marked tests
    # With --rungui: run ONLY gui-marked tests
    # NOTE: use get_closest_marker() instead of `"gui" in item.keywords`
    # because keywords includes parent package names (e.g. "gui" from
    # src/aydin/gui/), which would incorrectly match non-Qt tests.
    if not config.getoption("--rungui"):
        skip_gui = pytest.mark.skip(reason="need --rungui option to run")
        for item in items:
            if item.get_closest_marker("gui"):
                item.add_marker(skip_gui)
    else:
        skip_not_gui = pytest.mark.skip(reason="running only gui tests (--rungui)")
        for item in items:
            if not item.get_closest_marker("gui"):
                item.add_marker(skip_not_gui)
