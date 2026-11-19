"""Pytest configuration for the long-run stability regression.

This only adds test-side configuration; the sounddevice implementation
is untouched.  Everything can be steered with environment variables or
the ``--soak`` flag, so the same tests cover a quick CI run and a
multi-day soak (e.g. ``SD_LONGRUN_CYCLES=200000 pytest --soak``).
"""
import os

import pytest

SOAK_MARKER = "longrun"


def pytest_addoption(parser):
    parser.addoption(
        "--soak", action="store_true",
        help="run the extended long-soak profile (many more cycles)",
    )
    parser.addoption(
        "--soak-cycles", type=int, default=None,
        help="override the number of create/destroy cycles per test",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"{SOAK_MARKER}: repeated stream create/destroy long-run tests",
    )


def _int_env(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise pytest.UsageError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise pytest.UsageError(f"{name} must be positive, got {value}")
    return value


@pytest.fixture(scope="session")
def longrun_config(request):
    """Cycle counts and tolerance knobs for the regression."""
    soak = request.config.getoption("soak")
    cli_cycles = request.config.getoption("soak_cycles")
    if cli_cycles is not None:
        cycles = cli_cycles
    else:
        cycles = _int_env("SD_LONGRUN_CYCLES", 2000 if soak else 60)
    abort_cycles = _int_env(
        "SD_LONGRUN_ABORT_CYCLES", 1000 if soak else 30)
    # Resident memory is allowed a small, non-growing headroom for
    # allocator/arena noise.  Handles and threads must match exactly.
    rss_tolerance_kb = _int_env("SD_LONGRUN_RSS_TOL_KB", 2048)
    return {
        "cycles": cycles,
        "abort_cycles": abort_cycles,
        "blocks_per_session": _int_env("SD_LONGRUN_BLOCKS", 20),
        "rss_tolerance_kb": rss_tolerance_kb,
    }
