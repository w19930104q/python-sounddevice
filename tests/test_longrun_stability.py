"""Long-run stability regression for python-sounddevice.

Repeatedly creates and destroys real PortAudio sessions inside one
process and proves two things:

1. **Resources return to the starting point.**  Open file descriptors
   and thread counts must match exactly after every scenario; resident
   memory must stay within a fixed tolerance and show no monotonic
   upward trend.  This is what catches the "creeps up over a few days"
   class of regression.
2. **Samples stay identical to a one-shot session.**  A deterministic
   two-tone signal is played through a PulseAudio sink and recorded from
   its ``.monitor`` source; every repeated session must capture the same
   content as the single baseline session.

Both the normal teardown path (``stop`` + ``close``) and the
abnormal-interrupt path (``abort`` + ``close``, exceptions inside the
stream context, failed opens) are covered separately.

The sounddevice implementation is not modified.  Run normally with::

    pytest tests/

or as an extended soak with::

    pytest tests/ --soak
    SD_LONGRUN_CYCLES=200000 pytest tests/   # multi-day run

The tests need a PulseAudio backend exposing a sink/.monitor pair and
are skipped (not failed) when no loopback pair is available.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import sounddevice as sd

from regression_support import (
    DEFAULT_PRIME_BLOCKS,
    ResourceSnapshot,
    recordings_match,
    find_loopback_pair,
    make_test_signal,
    require_no_monotonic_growth,
    require_resource_return,
    run_duplex_session,
)

SAMPLERATE = 44100
BLOCKSIZE = 1024
# Buffer comfortably covers priming blocks plus the recorded region.
SIGNAL_FRAMES = (DEFAULT_PRIME_BLOCKS + 64) * BLOCKSIZE


@pytest.fixture(scope="session")
def loopback():
    try:
        in_device, out_device = find_loopback_pair()
    except (RuntimeError, OSError) as exc:
        pytest.skip(f"no loopback device available: {exc}")
    return in_device, out_device


@pytest.fixture(scope="session")
def signal():
    return make_test_signal(SAMPLERATE, SIGNAL_FRAMES)


def _warmup(loopback, signal, blocks):
    """One throwaway session so first-time backend setup doesn't count."""
    in_device, out_device = loopback
    run_duplex_session(in_device, out_device, SAMPLERATE, BLOCKSIZE,
                       blocks, signal)


def _report_trajectory(label, baseline, checkpoints):
    """Print the resource trajectory so a soak run is observable."""
    print(f"\n{label} resource trajectory "
          f"(baseline {baseline.as_dict()}):")
    for i, snap in enumerate(checkpoints):
        print(f"  checkpoint {i:2d}: {snap.as_dict()} "
              f"(rss {snap.rss_kb - baseline.rss_kb:+d} kB, "
              f"fds {snap.fds - baseline.fds:+d}, "
              f"threads {snap.threads - baseline.threads:+d})")


# ---------------------------------------------------------------------------
# Required: resources return to baseline over many normal create/destroy
# ---------------------------------------------------------------------------

@pytest.mark.longrun
def test_resources_return_to_baseline_repeated_sessions(
        loopback, signal, longrun_config):
    in_device, out_device = loopback
    cycles = longrun_config["cycles"]
    blocks = longrun_config["blocks_per_session"]
    tol_kb = longrun_config["rss_tolerance_kb"]

    _warmup(loopback, signal, blocks)
    baseline = ResourceSnapshot.capture()
    checkpoints = [baseline]
    checkpoint_every = max(1, cycles // 10)

    for cycle in range(1, cycles + 1):
        run_duplex_session(in_device, out_device, SAMPLERATE, BLOCKSIZE,
                           blocks, signal, abort=False)
        if cycle % checkpoint_every == 0 or cycle == cycles:
            snap = ResourceSnapshot.capture()
            checkpoints.append(snap)

    final = ResourceSnapshot.capture()
    _report_trajectory(f"normal sessions x{cycles}", baseline, checkpoints)
    require_resource_return(baseline, final, tol_kb,
                            label=f"normal sessions x{cycles}")
    require_no_monotonic_growth(checkpoints,
                                label=f"normal sessions x{cycles}")


# ---------------------------------------------------------------------------
# Required: samples from repeated sessions match a one-shot session
# ---------------------------------------------------------------------------

@pytest.mark.longrun
def test_repeated_session_samples_match_one_shot(
        loopback, signal, longrun_config):
    in_device, out_device = loopback
    blocks = longrun_config["blocks_per_session"]
    cycles = min(longrun_config["cycles"], 20)  # content check needs fewer

    # The "one-shot session": a single standalone stream lifecycle.
    reference = run_duplex_session(in_device, out_device, SAMPLERATE,
                                   BLOCKSIZE, blocks, signal)
    assert np.max(np.abs(reference)) > 1e-3, "baseline captured only silence"

    mismatches = []
    for cycle in range(cycles):
        candidate = run_duplex_session(in_device, out_device, SAMPLERATE,
                                       BLOCKSIZE, blocks, signal)
        ok, detail = recordings_match(reference, candidate)
        if not ok:
            mismatches.append((cycle, detail))
    assert not mismatches, f"session content diverged: {mismatches[:3]}"


def test_context_manager_session_samples(loopback, signal):
    """The documented ``with`` lifecycle yields the same content too."""
    in_device, out_device = loopback
    blocks = 16
    recorded = np.empty((blocks * BLOCKSIZE, 2), dtype="float32")
    # Mirror run_duplex_session: stream the continuous signal and skip
    # startup priming blocks, so the two paths capture equivalent audio.
    state = {"frames": 0, "seen": 0, "cursor": 0}
    total = len(signal)

    def callback(indata, outdata, frames, time_info, status):
        del time_info, status
        start = state["cursor"] % total
        end = start + frames
        if end <= total:
            outdata[:] = signal[start:end]
        else:
            head = total - start
            outdata[:head] = signal[start:]
            outdata[head:] = signal[: end - total]
        state["cursor"] += frames
        state["seen"] += 1
        if state["seen"] <= DEFAULT_PRIME_BLOCKS:
            return
        rec_end = min(state["frames"] + frames, len(recorded))
        recorded[state["frames"]:rec_end] = indata[: rec_end - state["frames"]]
        state["frames"] = rec_end

    with sd.Stream(samplerate=SAMPLERATE, blocksize=BLOCKSIZE,
                   device=(in_device, out_device), channels=(2, 2),
                   dtype="float32", callback=callback):
        while state["frames"] < len(recorded):
            sd.sleep(10)

    reference = run_duplex_session(in_device, out_device, SAMPLERATE,
                                   BLOCKSIZE, blocks, signal)
    ok, detail = recordings_match(reference, recorded)
    assert ok, detail


# ---------------------------------------------------------------------------
# Distinction required: abnormal-interrupt paths also release everything
# ---------------------------------------------------------------------------

@pytest.mark.longrun
def test_abort_path_resources_return_to_baseline(
        loopback, signal, longrun_config):
    """``abort()`` (immediate, buffers discarded) must not leak either."""
    in_device, out_device = loopback
    cycles = longrun_config["abort_cycles"]
    blocks = longrun_config["blocks_per_session"]
    tol_kb = longrun_config["rss_tolerance_kb"]

    _warmup(loopback, signal, blocks)
    baseline = ResourceSnapshot.capture()
    checkpoints = [baseline]
    checkpoint_every = max(1, cycles // 10)

    for cycle in range(1, cycles + 1):
        run_duplex_session(in_device, out_device, SAMPLERATE, BLOCKSIZE,
                           blocks, signal, abort=True)
        if cycle % checkpoint_every == 0 or cycle == cycles:
            checkpoints.append(ResourceSnapshot.capture())

    final = ResourceSnapshot.capture()
    _report_trajectory(f"aborted sessions x{cycles}", baseline, checkpoints)
    require_resource_return(baseline, final, tol_kb,
                            label=f"aborted sessions x{cycles}")
    require_no_monotonic_growth(checkpoints,
                                label=f"aborted sessions x{cycles}")


def test_aborted_session_still_records_signal(loopback, signal):
    """Abort only cuts the tail; the frames already captured are valid."""
    in_device, out_device = loopback
    recorded = run_duplex_session(in_device, out_device, SAMPLERATE,
                                  BLOCKSIZE, 20, signal, abort=True)
    rms = np.sqrt(np.mean(recorded ** 2, axis=0))
    assert np.max(rms) > 1e-3, f"aborted session lost its data: rms={rms}"


def test_exception_inside_context_releases_resources(loopback, signal):
    """An exception while a stream is active must still close it cleanly."""
    in_device, out_device = loopback
    baseline = ResourceSnapshot.capture()

    for _ in range(10):
        stream = None
        with pytest.raises(RuntimeError, match="simulated failure"):
            with sd.Stream(samplerate=SAMPLERATE, blocksize=BLOCKSIZE,
                           device=(in_device, out_device), channels=(2, 2),
                           dtype="float32") as stream:
                sd.sleep(100)
                raise RuntimeError("simulated failure")
        assert stream is not None and stream.closed

    final = ResourceSnapshot.capture()
    require_resource_return(baseline, final,
                            longrun_config_tol(), "exception-in-context x10")


def test_failed_open_does_not_leak_handles(loopback):
    """A rejected stream open must leave no descriptor behind."""
    baseline = ResourceSnapshot.capture()
    for _ in range(10):
        with pytest.raises(sd.PortAudioError):
            sd.InputStream(device=999, channels=2, samplerate=SAMPLERATE)
    final = ResourceSnapshot.capture()
    require_resource_return(baseline, final, longrun_config_tol(),
                            "failed-open x10")


def longrun_config_tol() -> int:
    """Tolerance for the short, fixed-cycle interruption tests."""
    return int(os.environ.get("SD_LONGRUN_RSS_TOL_KB", "2048"))


# ---------------------------------------------------------------------------
# Process-level check: Pa_Initialize/Pa_Terminate stay balanced
# ---------------------------------------------------------------------------

def test_portaudio_terminate_balances_initialize():
    """Each explicit initialize/terminate round must balance to zero.

    Run in a subprocess so the main test process' PortAudio lifetime is
    untouched.  A non-zero exit means the init/terminate refcount (or an
    assertion inside sounddevice) drifted.
    """
    code = (
        "import sounddevice as sd\n"
        "for _ in range(50):\n"
        "    sd._initialize()\n"
        "    sd._terminate()\n"
        "assert sd._initialized == 1, sd._initialized\n"
    )
    env = dict(os.environ)
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (
        f"init/terminate imbalance\nstdout:{result.stdout}\n"
        f"stderr:{result.stderr}")
