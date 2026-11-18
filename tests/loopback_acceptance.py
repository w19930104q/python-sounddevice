#!/usr/bin/env python3
"""Acceptance test for the hardened playrec() loopback path.

This runs hardware-free: the stream classes are replaced by simulated
streams that wire output channel k back to input channel k through a
fixed delay (a virtual loopback cable).  A fixed, deterministic test
signal is used, so results must be exactly reproducible between runs.

Covered scenarios:

1. Fixed-signal loopback: unified channel mapping, block-aligned
   playback/recording, exact and reproducible results.
2. Mapping unification rules of ``playrec(..., loopback=True)``.
3. Per-category overflow/underflow counts and classified errors
   (`CallbackStatusError`).
4. Non-loopback paths (`play()`, `rec()`, plain `playrec()`) behave
   exactly as before.

Run directly (``python3 tests/loopback_acceptance.py``) or with pytest.
"""

import contextlib
import ctypes.util
import os
import sys
import threading

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

# Allow importing sounddevice even if PortAudio is not installed
# system-wide (e.g. CI containers): fall back to a bundled/extracted
# libportaudio or to the PORTAUDIO_LIBRARY environment variable.
_orig_find_library = ctypes.util.find_library


def _find_library(name):
    result = _orig_find_library(name)
    if result is None and name == 'portaudio':
        candidates = [
            os.environ.get('PORTAUDIO_LIBRARY', ''),
            os.path.join(REPO_ROOT, 'src', '_sounddevice_data',
                         'portaudio-binaries', 'libportaudio.so.2'),
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
    return result


ctypes.util.find_library = _find_library

import numpy as np  # noqa: E402

import sounddevice as sd  # noqa: E402

SAMPLERATE = 48000
BLOCKSIZE = 256
FRAMES = 5000  # deliberately not a multiple of BLOCKSIZE
LATENCY = 300  # virtual loopback cable delay in frames


def fixed_signal(frames, channels=2, offset=0):
    """Deterministic multi-channel test signal (no randomness)."""
    t = (offset + np.arange(frames, dtype=np.float64)) / SAMPLERATE
    signal = np.empty((frames, channels), dtype=np.float32)
    for ch in range(channels):
        signal[:, ch] = (
            0.3 * np.sin(2 * np.pi * (997 + 131 * ch) * t + 0.5 * ch)
            + 0.1 * np.sign(np.sin(2 * np.pi * (113 + 37 * ch) * t)))
    return signal


class FakeStreamBase:
    """Common machinery for simulated streams.

    Mimics the parts of the Stream interface used by play()/rec()/
    playrec(): constructor keyword arguments, start(), stop(), close()
    and a finished_callback invoked from the callback thread.
    """

    def __init__(self, *, samplerate, channels, dtype, callback,
                 finished_callback, blocksize=0, **kwargs):
        self._callback = callback
        self._finished_callback = finished_callback
        self._blocksize = blocksize or BLOCKSIZE
        self.callback_count = 0
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, ignore_errors=True):
        pass

    def close(self, ignore_errors=True):
        if self._thread is not None:
            self._thread.join()

    def _run(self):
        try:
            self._produce()
        finally:
            self._finished_callback()

    def _status(self, script):
        status = sd.CallbackFlags()
        for name in script.get(self.callback_count, ()):
            setattr(status, name, True)
        return status


class SimulatedLoopbackStream(FakeStreamBase):
    """Simulated full-duplex stream with a virtual loopback cable.

    Output channel k appears on input channel k after exactly
    ``latency_frames`` frames.  ``status_script`` maps callback indices
    to iterables of CallbackFlags attribute names to be set.
    """

    def __init__(self, *, latency_frames=0, status_script=None, **kwargs):
        super().__init__(**kwargs)
        self._in_ch, self._out_ch = kwargs['channels']
        self._latency = latency_frames
        self._status_script = status_script or {}
        self._pending = np.zeros((self._latency, self._out_ch),
                                 dtype=np.float32)
        self.written_blocks = []

    def _produce(self):
        bs = self._blocksize
        while True:
            # Take bs frames from the wire (zero-padded before the first
            # played frame reaches the input):
            take = min(bs, len(self._pending))
            delayed = np.zeros((bs, self._out_ch), dtype=np.float32)
            if take:
                delayed[:take] = self._pending[:take]
                self._pending = self._pending[take:]
            indata = np.zeros((bs, self._in_ch), dtype=np.float32)
            n = min(self._in_ch, self._out_ch)
            indata[:, :n] = delayed[:, :n]
            outdata = np.zeros((bs, self._out_ch), dtype=np.float32)
            status = self._status(self._status_script)
            self.callback_count += 1
            try:
                self._callback(indata, outdata, bs, None, status)
            except sd.CallbackAbort:
                # paAbort: the final (silence-filled) block is discarded
                break
            self.written_blocks.append(outdata.copy())
            self._pending = np.concatenate([self._pending, outdata])


class SimulatedInputStream(FakeStreamBase):
    """Simulated input stream producing a fixed deterministic signal."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._channels = kwargs['channels']

    def _produce(self):
        bs = self._blocksize
        pos = 0
        while True:
            indata = fixed_signal(bs, self._channels, offset=pos)
            self.callback_count += 1
            try:
                self._callback(indata, bs, None, sd.CallbackFlags())
            except sd.CallbackAbort:
                break
            pos += bs


class SimulatedOutputStream(FakeStreamBase):
    """Simulated output stream collecting everything that is played."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._channels = kwargs['channels']
        self.written_blocks = []

    def _produce(self):
        bs = self._blocksize
        while True:
            outdata = np.zeros((bs, self._channels), dtype=np.float32)
            self.callback_count += 1
            try:
                self._callback(outdata, bs, None, sd.CallbackFlags())
            except sd.CallbackAbort:
                break
            self.written_blocks.append(outdata.copy())


class StreamFactory:
    """Binds fixed constructor arguments and records created streams."""

    def __init__(self, cls, **fixed_kwargs):
        self._cls = cls
        self._fixed_kwargs = fixed_kwargs
        self.instances = []

    def __call__(self, **kwargs):
        stream = self._cls(**dict(kwargs, **self._fixed_kwargs))
        self.instances.append(stream)
        return stream


@contextlib.contextmanager
def use_streams(stream=None, input_stream=None, output_stream=None):
    saved = sd.Stream, sd.InputStream, sd.OutputStream
    if stream is not None:
        sd.Stream = stream
    if input_stream is not None:
        sd.InputStream = input_stream
    if output_stream is not None:
        sd.OutputStream = output_stream
    try:
        yield
    finally:
        sd.Stream, sd.InputStream, sd.OutputStream = saved


def delayed(signal, latency):
    """Expected loopback recording: signal delayed by the cable latency."""
    out = np.zeros_like(signal)
    out[latency:] = signal[:len(signal) - latency]
    return out


def report(title, **diagnostics):
    print(f'--- {title} ---')
    for name, value in diagnostics.items():
        print(f'    {name}: {value}')


def zero_counts():
    return dict.fromkeys(('input_underflow', 'input_overflow',
                          'output_underflow', 'output_overflow',
                          'priming_output'), 0)


def test_loopback_fixed_signal_alignment_and_reproducibility():
    """Fixed signal through the hardened loopback path, run twice."""
    data = fixed_signal(FRAMES)
    recordings = []
    stats = []
    for run in (1, 2):
        factory = StreamFactory(SimulatedLoopbackStream,
                                latency_frames=LATENCY)
        with use_streams(stream=factory):
            out = sd.playrec(data, samplerate=SAMPLERATE,
                             blocksize=BLOCKSIZE, loopback=True,
                             blocking=True)
        stream = factory.instances[-1]
        recordings.append(out)
        stats.append((stream.callback_count, sd.get_status_counts()))

    expected = delayed(data, LATENCY)
    max_err = np.max(np.abs(recordings[0] - expected))
    report('loopback: fixed signal, mapping and block alignment',
           signal=f'fixed analytic signal, {FRAMES} frames, 2 channels',
           blocksize=BLOCKSIZE,
           callbacks=stats[0][0],
           cable_latency_frames=LATENCY,
           status='clean' if not sd.get_status() else str(sd.get_status()),
           status_counts=stats[0][1],
           max_abs_error_vs_expected=max_err,
           run1_equals_run2=np.array_equal(recordings[0], recordings[1]))

    assert max_err == 0.0, 'recorded signal is not the exactly delayed input'
    assert np.array_equal(recordings[0], recordings[1]), 'not reproducible'
    assert stats[0] == stats[1], 'diagnostics differ between runs'
    assert stats[0][1] == zero_counts()
    assert not sd.get_status()


def test_loopback_mapping_unification():
    """loopback=True unifies input/output channel mappings."""
    data = fixed_signal(FRAMES)

    # Conflicting explicit mappings are rejected:
    try:
        sd.playrec(data, samplerate=SAMPLERATE,
                   input_mapping=[1, 2], output_mapping=[2, 1],
                   loopback=True)
    except ValueError as e:
        assert 'identical' in str(e)
    else:
        raise AssertionError('conflicting mappings were not rejected')

    # A one-sided mapping is mirrored to the other direction; reversed
    # physical channel order must still yield column-aligned results:
    factory = StreamFactory(SimulatedLoopbackStream, latency_frames=LATENCY)
    with use_streams(stream=factory):
        out = sd.playrec(data, samplerate=SAMPLERATE, output_mapping=[2, 1],
                         blocksize=BLOCKSIZE, loopback=True, blocking=True)
    stream = factory.instances[-1]
    max_err = np.max(np.abs(out - delayed(data, LATENCY)))
    report('loopback: one-sided reversed mapping [2, 1]',
           stream_channels=f'in={stream._in_ch}, out={stream._out_ch}',
           max_abs_error_vs_expected=max_err)
    assert stream._in_ch == stream._out_ch == 2
    assert max_err == 0.0


def test_status_counts_and_classified_error():
    """Over-/underflows are counted per category and classified."""
    data = fixed_signal(FRAMES)
    script = {0: ('priming_output',),
              2: ('input_overflow',),
              3: ('output_underflow',),
              5: ('input_overflow',)}
    factory = StreamFactory(SimulatedLoopbackStream,
                            latency_frames=LATENCY, status_script=script)
    with use_streams(stream=factory):
        try:
            sd.playrec(data, samplerate=SAMPLERATE, blocksize=BLOCKSIZE,
                       loopback=True, blocking=True)
        except sd.CallbackStatusError as e:
            error = e
        else:
            raise AssertionError('strict loopback did not raise')

    counts = sd.get_status_counts()
    report('loopback: injected over-/underflows',
           injected=script,
           status_counts=counts,
           raised=type(error).__name__,
           error_message=str(error),
           error_counts_attribute=error.counts)

    assert counts == {'input_underflow': 0, 'input_overflow': 2,
                      'output_underflow': 1, 'output_overflow': 0,
                      'priming_output': 1}
    # priming_output is normal operation, not an error:
    assert error.counts == {'input_overflow': 2, 'output_underflow': 1}
    assert 'input overflow' in str(error)
    assert 'output underflow' in str(error)
    status = sd.get_status()
    assert status.input_overflow and status.output_underflow
    assert status.priming_output


def test_counts_without_strict_mode():
    """Plain playrec() records counts but never raises (unchanged)."""
    data = fixed_signal(FRAMES)
    script = {2: ('input_overflow',), 3: ('output_underflow',)}
    factory = StreamFactory(SimulatedLoopbackStream,
                            latency_frames=LATENCY, status_script=script)
    with use_streams(stream=factory):
        out = sd.playrec(data, samplerate=SAMPLERATE, channels=2,
                         blocksize=BLOCKSIZE, blocking=True)
    counts = sd.get_status_counts()
    report('plain playrec: injected over-/underflows (non-strict)',
           status_counts=counts,
           wait_returned=str(sd.get_status()))
    assert counts['input_overflow'] == 1
    assert counts['output_underflow'] == 1
    assert out.shape == data.shape


def test_non_loopback_paths_unchanged():
    """play()/rec()/plain playrec() behave exactly as before."""
    # rec(): the recorded data is exactly what the device delivered
    factory = StreamFactory(SimulatedInputStream)
    with use_streams(input_stream=factory):
        out = sd.rec(FRAMES, samplerate=SAMPLERATE, channels=2,
                     dtype='float32', blocksize=BLOCKSIZE, blocking=True)
    max_err = np.max(np.abs(out - fixed_signal(FRAMES)))
    report('rec: unchanged behavior',
           max_abs_error_vs_device_signal=max_err,
           status_counts=sd.get_status_counts())
    assert max_err == 0.0
    assert sd.get_status_counts() == zero_counts()

    # play(): the device receives exactly the data, then silence
    data = fixed_signal(FRAMES)
    factory = StreamFactory(SimulatedOutputStream)
    with use_streams(output_stream=factory):
        sd.play(data, samplerate=SAMPLERATE, blocksize=BLOCKSIZE,
                blocking=True)
    written = np.concatenate(factory.instances[-1].written_blocks)
    max_err = np.max(np.abs(written[:FRAMES] - data))
    tail_max = np.max(np.abs(written[FRAMES:]))
    report('play: unchanged behavior',
           written_frames=len(written),
           max_abs_error_vs_data=max_err,
           max_abs_tail_after_data=tail_max)
    assert max_err == 0.0
    assert tail_max == 0.0

    # plain playrec(): independent input/output mappings still work
    factory = StreamFactory(SimulatedLoopbackStream, latency_frames=LATENCY)
    with use_streams(stream=factory):
        out = sd.playrec(data, samplerate=SAMPLERATE, input_mapping=[1],
                         blocksize=BLOCKSIZE, blocking=True)
    max_err = np.max(np.abs(out[:, 0] - delayed(data, LATENCY)[:, 0]))
    report('plain playrec: independent mappings still allowed',
           out_channels=out.shape[1],
           max_abs_error_vs_expected=max_err)
    assert out.shape == (FRAMES, 1)
    assert max_err == 0.0


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith('test_')]
    print(f'running {len(tests)} acceptance scenarios '
          f'(samplerate={SAMPLERATE}, blocksize={BLOCKSIZE}, '
          f'frames={FRAMES})')
    for test in tests:
        test()
        print(f'PASS: {test.__name__}')
    print('all acceptance scenarios passed')


if __name__ == '__main__':
    main()
