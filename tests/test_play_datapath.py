"""Tests for the zero-copy playback data path of play()/playrec().

When dtype, memory layout and channel mapping of the playback data already
match what PortAudio expects, the user-provided array must be passed through
without copying (verified with ``numpy.shares_memory()``).  Only mismatched
data is converted; non-contiguous arrays take an explicit, one-time
conversion path.  In all cases the audio sent to PortAudio must be
sample-identical to the pre-optimization behavior.
"""

import numpy as np
import pytest

import sounddevice as sd


def make_ctx(data, mapping=None, loop=False):
    """Run the non-realtime part of play() and return the context."""
    ctx = sd._CallbackContext(loop=loop)
    ctx.frames = ctx.check_data(data, mapping, device=None)
    return ctx


def simulate_playback(ctx, blocksize, nblocks):
    """Drive *ctx* exactly like play()'s audio callback does.

    Returns the concatenation of all *outdata* buffers that would have
    been handed to PortAudio.  Stops early if the callback aborts
    (which happens when non-looped playback is finished).
    """
    chunks = []
    for _ in range(nblocks):
        outdata = np.empty((blocksize, ctx.output_channels),
                           dtype=ctx.output_dtype)
        ctx.callback_enter(sd.CallbackFlags(), outdata)
        ctx.write_outdata(outdata)
        chunks.append(outdata)
        try:
            ctx.callback_exit()
        except sd.CallbackAbort:
            break
    return np.concatenate(chunks, axis=0)


def expected_output(data, mapping, blocksize, nblocks, loop=False):
    """Independently compute the expected PortAudio input sample-by-sample."""
    data = np.asarray(data)
    if data.ndim < 2:
        data = data.reshape(-1, 1)
    frames = data.shape[0]
    dtype = sd._check_dtype(data.dtype)
    mapping, channels = sd._check_mapping(mapping, data.shape[1])
    silent_channels = np.setdiff1d(np.arange(channels), mapping)
    silence = 128 if dtype == 'uint8' else 0
    # NumPy assignment casting, as used in the audio callback:
    src = data.astype(np.dtype(dtype))
    total = nblocks * blocksize
    out = np.full((total, channels), silence, dtype=dtype)
    if loop:
        src = np.tile(src, (-(-total // frames), 1))[:total]
        out[:, mapping] = src
    else:
        out[:min(frames, total), mapping] = src[:total]
    out[:, silent_channels] = silence
    return out


def check_playback(data, mapping=None, blocksize=64, loop=False, nblocks=20):
    """Assert sample-identical output of the optimized data path."""
    ctx = make_ctx(data, mapping, loop=loop)
    result = simulate_playback(ctx, blocksize, nblocks)
    expected = expected_output(data, mapping, blocksize,
                               len(result) // blocksize, loop=loop)
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result, expected)
    return ctx


# --- zero-copy passthrough (shares_memory assertions) ---------------------

def test_matching_2d_array_is_not_copied():
    data = np.random.randn(1000, 2).astype('float32')
    ctx = make_ctx(data)
    assert np.shares_memory(ctx.data, data)
    assert ctx._passthrough


def test_matching_1d_array_is_not_copied():
    data = np.random.randn(1000).astype('float32')
    ctx = make_ctx(data)
    assert np.shares_memory(ctx.data, data)
    assert ctx._passthrough


@pytest.mark.parametrize('dtype', ['float32', 'int32', 'int16', 'int8',
                                   'uint8'])
def test_all_native_dtypes_are_not_copied(dtype):
    data = np.zeros((1000, 2), dtype=dtype)
    ctx = make_ctx(data)
    assert np.shares_memory(ctx.data, data)
    assert ctx._passthrough
    assert ctx.output_dtype == dtype


def test_float64_is_not_copied_but_not_passthrough():
    # float64 must be converted to float32, but block-by-block in the
    # callback -- the whole array must not be copied up-front.
    data = np.random.randn(1000, 2)
    ctx = make_ctx(data)
    assert np.shares_memory(ctx.data, data)
    assert not ctx._passthrough
    assert ctx.output_dtype == 'float32'


# --- explicit conversion path for non-contiguous data ---------------------

def test_strided_array_takes_explicit_conversion_path():
    base = np.random.randn(2000, 2).astype('float32')
    data = base[::2]
    assert not data.flags.c_contiguous
    ctx = make_ctx(data)
    assert not np.shares_memory(ctx.data, data)
    assert ctx.data.flags.c_contiguous
    np.testing.assert_array_equal(ctx.data, data)
    assert ctx._passthrough  # contiguous after conversion


def test_fortran_order_array_takes_explicit_conversion_path():
    data = np.asfortranarray(np.random.randn(1000, 2).astype('float32'))
    assert not data.flags.c_contiguous
    ctx = make_ctx(data)
    assert not np.shares_memory(ctx.data, data)
    assert ctx.data.flags.c_contiguous
    np.testing.assert_array_equal(ctx.data, data)


def test_transposed_array_takes_explicit_conversion_path():
    data = np.random.randn(2, 1000).astype('float32').T
    assert not data.flags.c_contiguous
    ctx = make_ctx(data)
    assert not np.shares_memory(ctx.data, data)
    assert ctx.data.flags.c_contiguous
    np.testing.assert_array_equal(ctx.data, data)


# --- sample-identical output ----------------------------------------------

@pytest.mark.parametrize('dtype', ['float64', 'float32', 'int32', 'int16',
                                   'int8', 'uint8'])
def test_output_sample_identical_dtypes(dtype):
    if np.dtype(dtype).kind == 'f':
        data = np.random.randn(1000, 2).astype(dtype)
    else:
        data = np.random.randint(-100, 100, (1000, 2)).astype(dtype)
    check_playback(data, blocksize=64)


def test_output_sample_identical_mono():
    data = np.random.randn(1000).astype('float32')
    check_playback(data, blocksize=64)


def test_output_sample_identical_mapping_swapped():
    data = np.random.randn(1000, 2).astype('float32')
    ctx = check_playback(data, mapping=[2, 1], blocksize=64)
    assert not ctx._passthrough


def test_output_sample_identical_mapping_with_silent_channel():
    data = np.random.randn(1000).astype('float32')
    ctx = check_playback(data, mapping=[2], blocksize=64)
    assert not ctx._passthrough
    assert list(ctx.silent_channels) == [0]


def test_output_sample_identical_uint8_silence_padding():
    # 1000 frames with blocksize 64: the final partial block and the
    # aborted block must be padded with the uint8 silence value (128).
    data = np.random.randint(0, 255, (1000, 2)).astype('uint8')
    ctx = make_ctx(data)
    result = simulate_playback(ctx, blocksize=64, nblocks=100)
    assert np.all(result[1000:] == 128)


def test_output_sample_identical_noncontiguous():
    base = np.random.randn(2000, 2).astype('float32')
    check_playback(base[::2], blocksize=64)
    check_playback(np.asfortranarray(base[:1000]), blocksize=64)


def test_output_sample_identical_loop():
    data = np.random.randn(300, 2).astype('float32')
    ctx = check_playback(data, blocksize=64, loop=True, nblocks=10)
    assert ctx._passthrough


def test_output_sample_identical_loop_noncontiguous():
    data = np.random.randn(600, 2).astype('float32')[::2]
    check_playback(data, blocksize=64, loop=True, nblocks=10)


# --- short playback acceptance on a real device ----------------------------

PLAYBACK_TIMEOUT = 10  # seconds


def _play_with_timeout(data, samplerate, check=None, **kwargs):
    """Play *data*, skipping if no usable output device is available.

    *check* (if given) is called with the callback context while the
    stream is (presumably) still running.
    """
    try:
        sd.query_devices(device=None, kind='output')
    except Exception as e:
        pytest.skip(f'no output device available: {e}')
    sd.play(data, samplerate, **kwargs)
    ctx = sd._last_callback
    try:
        if check is not None:
            check(ctx)
        if not ctx.event.wait(PLAYBACK_TIMEOUT):
            pytest.skip('output device did not finish playback in time')
    finally:
        sd.stop()
    return ctx


def test_short_playback_passthrough():
    samplerate = 48000
    t = np.arange(int(samplerate * 0.25)) / samplerate
    data = (0.2 * np.sin(2 * np.pi * 440 * t)).astype('float32')

    def check(ctx):
        # The playback buffer must be passed through without copying:
        assert np.shares_memory(ctx.data, data)
        assert ctx._passthrough

    _play_with_timeout(data, samplerate, check=check)


def test_short_playback_float64():
    samplerate = 48000
    t = np.arange(int(samplerate * 0.25)) / samplerate
    data = 0.2 * np.sin(2 * np.pi * 440 * t)  # float64

    def check(ctx):
        assert ctx.output_dtype == 'float32'
        assert np.shares_memory(ctx.data, data)  # still no up-front copy

    _play_with_timeout(data, samplerate, check=check)


def test_short_playback_noncontiguous():
    samplerate = 48000
    t = np.arange(int(samplerate * 0.5)) / samplerate
    data = (0.2 * np.sin(2 * np.pi * 440 * t)).astype('float32')[::2]

    def check(ctx):
        assert ctx.data.flags.c_contiguous
        assert not np.shares_memory(ctx.data, data)

    _play_with_timeout(data, samplerate, check=check)


def test_short_playback_multichannel_mapping():
    samplerate = 48000
    t = np.arange(int(samplerate * 0.25)) / samplerate
    data = np.column_stack([0.2 * np.sin(2 * np.pi * 440 * t),
                            0.2 * np.sin(2 * np.pi * 550 * t)]).astype(
        'float32')
    _play_with_timeout(data, samplerate, mapping=[2, 1])
