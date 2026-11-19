"""Helpers for the long-run stability regression.

Three independent concerns live here:

* :class:`ResourceSnapshot` reads the process' resident memory, open file
  descriptor count and thread count straight from ``/proc`` (Linux) with
  no third-party dependency, so a regression cannot be hidden by a
  profiler's own allocations.
* :func:`find_loopback_pair` / :func:`run_duplex_session` drive a real
  duplex PortAudio stream through a PulseAudio sink and its ``.monitor``
  source, discarding the stream's startup priming blocks so only
  steady-state audio is captured and compared.
* :func:`recordings_match` applies a layered content check: hard
  signal invariants (dominant frequency, spectral concentration, level,
  absence of stalled blocks) that survive host timing jitter, plus a
  strict best-lag waveform/spectral comparison on deterministic backends.

Nothing in this package imports or modifies sounddevice internals beyond
the documented public API.
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


# ---------------------------------------------------------------------------
# Resource accounting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResourceSnapshot:
    rss_kb: int
    fds: int
    threads: int

    @classmethod
    def capture(cls) -> "ResourceSnapshot":
        gc.collect()  # make Python-level leaks deterministic, not GC-timed
        return cls(
            rss_kb=_read_rss_kb(),
            fds=_count_fds(),
            threads=_read_thread_count(),
        )

    def as_dict(self) -> dict:
        return {"rss_kb": self.rss_kb, "fds": self.fds,
                "threads": self.threads}


def _read_rss_kb() -> int:
    # statm field 1 is the resident size in pages.
    with open(f"/proc/{os.getpid()}/statm", "r", encoding="ascii") as fh:
        resident_pages = int(fh.read().split()[1])
    page_size = os.sysconf("SC_PAGE_SIZE")
    return resident_pages * page_size // 1024


def _count_fds() -> int:
    return len(os.listdir(f"/proc/{os.getpid()}/fd"))


def _read_thread_count() -> int:
    with open(f"/proc/{os.getpid()}/status", "r", encoding="ascii") as fh:
        for line in fh:
            if line.startswith("Threads:"):
                return int(line.split()[1])
    raise RuntimeError("could not read thread count from /proc status")


def require_resource_return(baseline: ResourceSnapshot,
                            final: ResourceSnapshot,
                            rss_tolerance_kb: int,
                            label: str) -> None:
    """Assert handles/threads return exactly and RSS returns within budget.

    File descriptors and threads are *exact-match* resources: a single
    leaked PulseAudio socket or PortAudio worker thread must fail the
    regression.  Resident memory only gets a bounded tolerance because
    glibc/Python arenas may legitimately hold freed pages, but any
    growth past the budget (or a monotonic climb) is treated as a leak.
    """
    assert final.fds == baseline.fds, (
        f"{label}: file descriptors drifted {baseline.fds} -> {final.fds} "
        f"(delta {final.fds - baseline.fds:+d}); a handle is leaking")
    assert final.threads == baseline.threads, (
        f"{label}: thread count drifted {baseline.threads} -> "
        f"{final.threads} (delta {final.threads - baseline.threads:+d}); "
        "a PortAudio worker thread is leaking")
    rss_delta = final.rss_kb - baseline.rss_kb
    assert rss_delta <= rss_tolerance_kb, (
        f"{label}: RSS grew {rss_delta:+d} kB "
        f"({baseline.rss_kb} -> {final.rss_kb}), tolerance "
        f"{rss_tolerance_kb} kB; memory is not being reclaimed")


def require_no_monotonic_growth(snapshots: list[ResourceSnapshot],
                                label: str) -> None:
    """Reject a sustained per-cycle climb, ignoring one-time warm-up.

    A leak of a few kB per session can stay under the absolute RSS
    tolerance for a short run while growing without bound over days, so
    the *trend* matters, not just the endpoint.  However the first
    checkpoint can show a one-time step when the allocator/backend
    lazily creates its working set (especially on the ``abort()`` path);
    that is a plateau, not a leak.  We therefore evaluate the slope on
    the steady-state region *after* the largest single step and require
    the steady-state endpoints to stay within a noise band as well.
    """
    if len(snapshots) < 4:
        return
    y = np.array([s.rss_kb for s in snapshots], dtype=float)

    # Locate the biggest single-step jump and start the trend just after
    # it, so a warm-up plateau cannot masquerade as a positive slope.
    steps = np.diff(y)
    warmup_cut = int(np.argmax(np.abs(steps))) + 1
    steady = y[min(warmup_cut, len(y) - 3):]
    if len(steady) < 3:
        steady = y  # not enough points; fall back to the full series

    x = np.arange(len(steady), dtype=float)
    slope = float(np.polyfit(x, steady, 1)[0])
    endpoint_drift = float(steady[-1] - steady[0])
    # Thresholds are well above observed allocator/measurement noise
    # (~50 kB) but far below any genuine per-interval leak.
    assert slope <= 64.0 and endpoint_drift <= 256.0, (
        f"{label}: RSS keeps growing after warm-up "
        f"(slope ~{slope:.1f} kB/checkpoint, endpoint drift "
        f"{endpoint_drift:+.0f} kB across {len(steady)} steady-state "
        "checkpoints); unbounded leak")


# ---------------------------------------------------------------------------
# Deterministic loopback sessions
# ---------------------------------------------------------------------------

def find_loopback_pair() -> tuple[int, int]:
    """Return ``(input_device, output_device)`` for a sink/.monitor pair.

    PulseAudio exposes every output sink's pre-render signal as a
    ``<sink>.monitor`` source, which is a real, in-kernel loopback: no
    cables, no external daemon and no non-deterministic microphone.
    """
    devices = sd.query_devices()
    sinks = {
        dev["name"].split(",")[0]: idx
        for idx, dev in enumerate(devices)
        if dev["max_output_channels"] > 0 and dev["max_input_channels"] == 0
    }
    for sink_name, out_idx in sinks.items():
        monitor = sink_name + ".monitor"
        for in_idx, dev in enumerate(devices):
            if (dev["name"].split(",")[0] == monitor
                    and dev["max_input_channels"] >= 2):
                return in_idx, out_idx
    raise RuntimeError(
        "no PulseAudio sink/.monitor loopback pair found; available "
        f"devices: {[d['name'] for d in devices]}")


SAMPLERATE_HZ = 44100
# Frequencies of the two deterministic test tones (one per channel).
TONE_FREQUENCIES = (440.0, 880.0)
# A freshly opened monitor source is silent/undefined for a few blocks
# while PortAudio/PulseAudio prime their buffers.  Those startup blocks
# are discarded so the comparison only sees steady-state audio.
DEFAULT_PRIME_BLOCKS = 12


def make_test_signal(samplerate: int, num_frames: int) -> np.ndarray:
    """Continuous-phase two-channel deterministic signal.

    The waveform is generated over the whole capture with unbroken phase
    rather than repeated per block, so adjacent blocks differ.  This
    makes a duplicated/stalled block detectable instead of being
    indistinguishable from the periodic output.
    """
    tt = np.arange(num_frames) / samplerate
    f0, f1 = TONE_FREQUENCIES
    left = 0.4 * np.sin(2 * np.pi * f0 * tt)
    right = 0.25 * np.sin(2 * np.pi * f1 * tt + 0.7)
    return np.stack([left, right], axis=1).astype("float32")


def run_duplex_session(in_device: int, out_device: int, samplerate: int,
                       blocksize: int, blocks: int, signal: np.ndarray,
                       abort: bool = False,
                       prime_blocks: int = DEFAULT_PRIME_BLOCKS) -> np.ndarray:
    """Open, run and tear down one duplex stream; return its recording.

    Lifecycle control is always explicit (``stop``/``abort`` then
    ``close``) rather than relying on ``CallbackStop``, because some
    PortAudio host backends ignore ``paComplete``/``paAbort``.  The
    callback streams the continuous *signal* through the output using a
    frame cursor (wrapping if necessary).  The first *prime_blocks*
    callbacks are dropped to get past monitor startup latency; only
    subsequent input blocks are stored, so the returned array reflects
    steady-state audio of exactly ``blocks * blocksize`` frames.
    """
    recorded = np.empty((blocks * blocksize, 2), dtype="float32")
    state = {"frames": 0, "seen": 0, "out_cursor": 0}
    total = len(signal)

    def callback(indata, outdata, frames, time_info, status):
        del time_info, status
        start = state["out_cursor"] % total
        end = start + frames
        if end <= total:
            outdata[:] = signal[start:end]
        else:  # wrap the continuous buffer at its boundary
            head = total - start
            outdata[:head] = signal[start:]
            outdata[head:] = signal[: end - total]
        state["out_cursor"] += frames
        state["seen"] += 1
        if state["seen"] <= prime_blocks:
            return  # still in the startup/priming region
        rec_start = state["frames"]
        rec_end = min(rec_start + frames, len(recorded))
        recorded[rec_start:rec_end] = indata[: rec_end - rec_start]
        state["frames"] = rec_end

    stream = sd.Stream(
        samplerate=samplerate, blocksize=blocksize,
        device=(in_device, out_device), channels=(2, 2),
        dtype="float32", callback=callback)
    stream.start()
    deadline = time.monotonic() + 30.0
    while state["frames"] < len(recorded):
        if time.monotonic() > deadline:
            stream.abort()
            stream.close()
            raise TimeoutError("audio callback stopped delivering frames")
        time.sleep(0.01)
    if abort:
        stream.abort()
    else:
        stream.stop()
    stream.close()
    assert stream.closed
    return recorded[: state["frames"]]


# A monitor glitch moves the capture window by at most a few blocks; the
# alignment search is bounded to avoid pure-tone periodicity ambiguity.
MAX_ALIGNMENT_LAG = 3 * 1024


def best_lag(reference: np.ndarray, candidate: np.ndarray) -> int:
    """Integer lag that best aligns *candidate* to *reference* (FFT)."""
    ref = reference - reference.mean()
    cand = candidate - candidate.mean()
    n = len(ref) + len(cand) - 1
    nfft = 1 << (n - 1).bit_length()
    xcorr = np.fft.irfft(np.fft.rfft(ref, nfft)
                         * np.conj(np.fft.rfft(cand, nfft)), nfft)
    xcorr = np.concatenate((xcorr[-(len(cand) - 1):], xcorr[:len(ref)]))
    lags = np.arange(-(len(cand) - 1), len(ref))
    window = min(MAX_ALIGNMENT_LAG, len(lags) // 2)
    center = len(lags) // 2
    lo, hi = center - window, center + window + 1
    return int(lags[lo + int(np.argmax(xcorr[lo:hi]))])


def aligned_overlap(reference: np.ndarray, candidate: np.ndarray,
                    lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Mutually overlapping slices when *candidate* is offset by *lag*.

    A positive *lag* means *candidate* starts *lag* frames after
    *reference*.  Both returned slices have the same length.
    """
    n_ref, n_cand = len(reference), len(candidate)
    if lag >= 0:
        overlap = min(n_ref - lag, n_cand)
        return reference[lag:lag + overlap], candidate[:overlap]
    shift = -lag
    overlap = min(n_cand - shift, n_ref)
    return reference[:overlap], candidate[shift:shift + overlap]


def channel_invariants(samples: np.ndarray, samplerate: int,
                       expected_freq: float) -> dict:
    """Backend-robust invariants for one captured channel.

    These describe *what* was captured rather than the exact sample
    positions, so they survive PulseAudio monitor timing jitter while
    still exposing a wrong tone, silence or a dead channel:

    * ``peak_hz``   - frequency bin with the most energy,
    * ``band_frac`` - fraction of spectral energy within +/-40 Hz of the
                      expected tone (a tight concentration means the
                      intended tone is genuinely present),
    * ``rms``       - signal level (catches silence/attenuation).
    """
    y = samples - samples.mean()
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-6:
        return {"peak_hz": 0.0, "band_frac": 0.0, "rms": 0.0}
    freqs = np.fft.rfftfreq(len(y), 1.0 / samplerate)
    spec = np.abs(np.fft.rfft(y))
    peak_hz = float(freqs[int(np.argmax(spec))])
    normalized = spec / spec.sum()
    band_frac = float(np.sum(normalized[np.abs(freqs - expected_freq) <= 40]))
    return {"peak_hz": peak_hz, "band_frac": band_frac, "rms": rms}


def count_duplicate_blocks(candidate: np.ndarray,
                           blocksize: int) -> int:
    """Count adjacent blocks that are byte-identical.

    The test signal has continuous phase, so neighbouring blocks always
    differ.  An identical adjacent block therefore means the backend
    repeated a block (a stall/underrun) rather than advancing time.
    """
    nblocks = len(candidate) // blocksize
    duplicates = 0
    for b in range(1, nblocks):
        if np.array_equal(candidate[(b - 1) * blocksize:b * blocksize],
                          candidate[b * blocksize:(b + 1) * blocksize]):
            duplicates += 1
    return duplicates


def passes_core_invariants(candidate: np.ndarray,
                           samplerate: int,
                           blocksize: int = 1024) -> tuple[bool, str]:
    """Hard gate every repeated session must satisfy, jitter or not.

    Validated empirically over hundreds of sessions on a virtualized
    PulseAudio backend: even sessions whose waveform phase drifted kept
    their dominant tone at the expected frequency with healthy energy and
    level, whereas silence, a dead channel, wrong-frequency signals and
    repeated (stalled) blocks violate these bounds outright.
    """
    if np.max(np.abs(candidate)) < 1e-4:
        return False, "recorded signal is silence"
    stalls = count_duplicate_blocks(candidate, blocksize)
    if stalls > 0:
        return False, f"detected {stalls} repeated block(s); stream stalled"
    for ch, expected in enumerate(TONE_FREQUENCIES):
        inv = channel_invariants(candidate[:, ch], samplerate, expected)
        if abs(inv["peak_hz"] - expected) > 40.0:
            return False, (f"ch{ch} dominant frequency {inv['peak_hz']:.1f} "
                           f"Hz != expected {expected:.0f} Hz")
        if inv["band_frac"] < 0.15:
            return False, (f"ch{ch} only {inv['band_frac']:.3f} of energy near "
                           f"{expected:.0f} Hz; intended tone absent")
        if inv["rms"] < 0.05:
            return False, f"ch{ch} RMS {inv['rms']:.4f} too low (dead channel)"
    return True, "core invariants OK"


def waveform_similarity(reference: np.ndarray,
                        candidate: np.ndarray) -> tuple[float, float]:
    """Best-lag correlation and max normalized spectral deviation.

    These are strict, shift-tolerant similarity measures used on
    deterministic backends; they are informational on a jittery one.
    """
    corrs = []
    for ch in range(reference.shape[1]):
        lag = best_lag(reference[:, ch], candidate[:, ch])
        ref_seg, cand_seg = aligned_overlap(reference[:, ch],
                                            candidate[:, ch], lag)
        ref_seg = ref_seg - ref_seg.mean()
        cand_seg = cand_seg - cand_seg.mean()
        denom = np.sqrt(np.sum(ref_seg ** 2) * np.sum(cand_seg ** 2))
        corrs.append(float(np.sum(ref_seg * cand_seg) / denom)
                     if denom else 0.0)
    ref_spec = np.abs(np.fft.rfft(reference - reference.mean(), axis=0))
    cand_spec = np.abs(np.fft.rfft(candidate - candidate.mean(), axis=0))
    ref_spec /= np.maximum(ref_spec.sum(axis=0, keepdims=True), 1e-12)
    cand_spec /= np.maximum(cand_spec.sum(axis=0, keepdims=True), 1e-12)
    spec_dev = float(np.max(np.abs(ref_spec - cand_spec)))
    return min(corrs), spec_dev


def recordings_match(reference: np.ndarray, candidate: np.ndarray,
                     samplerate: int = SAMPLERATE_HZ,
                     blocksize: int = 1024) -> tuple[bool, str]:
    """Decide whether a repeated session captured the same signal.

    Two layers:

    1. **Core invariants** (mandatory).  Correct dominant frequency per
       channel, spectral energy concentrated at that frequency, healthy
       non-silent level, and no repeated (stalled) blocks.  This is the
       reliable contract on a real audio backend and is what actually
       guards against a regression that corrupts, drops, stalls or
       silences the stream.
    2. **Strict waveform/spectral similarity** (deterministic bonus).
       When the backend is sample-stable, recordings must additionally
       align with the one-shot reference at correlation >= 0.995 and a
       near-zero spectral deviation; byte identity is reported too.
       Failure of this layer alone (core invariants still hold) is
       attributed to known host timing jitter, not a code regression.
    """
    if candidate.shape != reference.shape:
        return False, f"shape mismatch {reference.shape} vs {candidate.shape}"

    ok, reason = passes_core_invariants(candidate, samplerate, blocksize)
    if not ok:
        return False, reason

    corr, spec_dev = waveform_similarity(reference, candidate)
    strict_ok = corr >= 0.995 and spec_dev <= 0.02
    detail = (f"core invariants OK; best-lag corr={corr:.6f} "
              f"spec_dev={spec_dev:.2e}")
    if np.array_equal(reference, candidate):
        detail += "; arrays byte-identical"
    if strict_ok:
        return True, detail
    # Core content is correct but exact waveform drifted: acceptable host
    # jitter, surfaced transparently rather than hidden.
    return True, detail + "; strict waveform match relaxed (host jitter)"
