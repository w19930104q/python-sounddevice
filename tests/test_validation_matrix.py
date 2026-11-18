"""Acceptance tests for the validation/error-classification matrix.

See doc/api/validation-matrix.rst for the full matrix.  Each test ID
encodes one matrix cell: ``<entry-point>-<aspect>-<case>``.

Entry points (rows): play(), rec(), playrec(), Stream (incl.
InputStream/OutputStream).

Aspects (columns): dtype, shape, channels, mapping.

All invalid combinations must raise a classified error
(`StreamValidationError` subclass, or the documented `TypeError` for
non-integer frames/channels in rec()) *before* the stream is opened.
Valid combinations must behave as before: they never raise
`StreamValidationError`; on machines without audio hardware they fail
later with `PortAudioError` when the stream is opened.
"""

import numpy as np
import pytest

import sounddevice as sd


@pytest.fixture
def fake_device(monkeypatch):
    """Stub device queries so tests don't depend on audio hardware."""
    info = {
        'name': 'fake device',
        'index': 0,
        'hostapi': 0,
        'max_input_channels': 2,
        'max_output_channels': 2,
        'default_low_input_latency': 0.01,
        'default_low_output_latency': 0.01,
        'default_high_input_latency': 0.1,
        'default_high_output_latency': 0.1,
        'default_samplerate': 44100.0,
    }
    monkeypatch.setattr(sd, 'query_devices', lambda *a, **k: info)
    monkeypatch.setattr(sd, '_get_device_id', lambda *a, **k: 0)
    return info


@pytest.fixture
def forbid_open_stream(monkeypatch):
    """Fail if Pa_OpenStream is reached despite invalid input."""

    class Guard:
        """Delegate everything to the real lib except Pa_OpenStream."""

        def __init__(self, lib):
            object.__setattr__(self, '_lib', lib)

        def __getattr__(self, name):
            if name == 'Pa_OpenStream':
                raise AssertionError(
                    'Pa_OpenStream was called although validation '
                    'should have rejected the input beforehand')
            return getattr(self._lib, name)

    monkeypatch.setattr(sd, '_lib', Guard(sd._lib))


def check_invalid(thunk, error, match=None):
    with pytest.raises(error, match=match):
        thunk()


def check_valid(thunk):
    """Run a valid combination.

    Must not raise StreamValidationError.  On machines without audio
    hardware, opening the stream fails with PortAudioError -- which
    still proves that validation passed.  On machines with hardware
    the stream may actually open; it is closed again immediately.
    """
    try:
        result = thunk()
    except sd.PortAudioError:
        pass  # No usable audio device: validation passed, open failed
    else:
        if isinstance(result, sd._StreamBase):
            result.close()
    finally:
        sd.stop()


# --- play() ------------------------------------------------------------

PLAY_INVALID = [
    pytest.param(lambda: sd.play(np.zeros((8, 2), dtype='complex64')),
                 sd.DtypeError, 'Unsupported data type',
                 id='play-dtype-unsupported'),
    pytest.param(lambda: sd.play(np.zeros((8, 2), dtype='float16')),
                 sd.DtypeError, 'Unsupported data type',
                 id='play-dtype-float16'),
    pytest.param(lambda: sd.play(np.zeros((2, 2, 2))),
                 sd.ShapeError, 'one- or two-dimensional',
                 id='play-shape-3d'),
    pytest.param(lambda: sd.play(np.zeros((8, 1)), mapping=[0]),
                 sd.MappingError, 'must not be < 1',
                 id='play-mapping-below-1'),
    pytest.param(lambda: sd.play(np.zeros((8, 1)), mapping=[1.5]),
                 sd.MappingError, 'must contain integers',
                 id='play-mapping-non-integer'),
    pytest.param(lambda: sd.play(np.zeros((8, 1)), mapping=[]),
                 sd.MappingError, 'must not be empty',
                 id='play-mapping-empty'),
    pytest.param(lambda: sd.play(np.zeros((8, 2)), mapping=[1, 1]),
                 sd.MappingError, 'only appear once',
                 id='play-mapping-duplicate'),
    pytest.param(lambda: sd.play(np.zeros((8, 2)), mapping=[1, 2, 3]),
                 sd.MappingError, 'size of output mapping',
                 id='play-mapping-length-mismatch'),
]

PLAY_VALID = [
    pytest.param(lambda: sd.play(np.zeros((8, 1))),
                 id='play-valid-mono'),
    pytest.param(lambda: sd.play(np.zeros((8, 2), dtype='float64')),
                 id='play-valid-float64-converted'),
    pytest.param(lambda: sd.play(np.zeros((8, 2)), mapping=[2, 1]),
                 id='play-valid-mapping-swapped'),
    pytest.param(lambda: sd.play(np.zeros((8, 1)), mapping=[1]),
                 id='play-valid-mapping-mono-explicit'),
]

# --- rec() -------------------------------------------------------------

REC_INVALID = [
    pytest.param(lambda: sd.rec(8, channels=1, dtype='complex64'),
                 sd.DtypeError, 'Unsupported data type',
                 id='rec-dtype-unsupported'),
    pytest.param(lambda: sd.rec(8, channels=1, dtype='not-a-dtype'),
                 sd.DtypeError, 'Unsupported dtype',
                 id='rec-dtype-unknown'),
    pytest.param(lambda: sd.rec(8, out=np.zeros((8, 1), dtype='float16')),
                 sd.DtypeError, 'Unsupported data type',
                 id='rec-dtype-out'),
    pytest.param(lambda: sd.rec(8, out=np.zeros(8)),
                 sd.ShapeError, 'two-dimensional',
                 id='rec-shape-out-1d'),
    pytest.param(lambda: sd.rec(-1, channels=1),
                 sd.ShapeError, 'Invalid shape',
                 id='rec-shape-frames-negative'),
    pytest.param(lambda: sd.rec(8, channels=0),
                 sd.ChannelsError, 'at least 1',
                 id='rec-channels-zero'),
    pytest.param(lambda: sd.rec(8, channels=1.5),
                 TypeError, "'channels' must be an integer",
                 id='rec-channels-non-integer'),
    pytest.param(lambda: sd.rec('a lot', channels=1),
                 TypeError, "'frames' must be an integer",
                 id='rec-frames-non-integer'),
    pytest.param(lambda: sd.rec(8, mapping=[0]),
                 sd.MappingError, 'must not be < 1',
                 id='rec-mapping-below-1'),
    pytest.param(lambda: sd.rec(8, out=np.zeros((8, 2)), mapping=[1]),
                 sd.MappingError, 'size of input mapping',
                 id='rec-mapping-length-mismatch'),
]

REC_VALID = [
    pytest.param(lambda: sd.rec(8, channels=1),
                 id='rec-valid-default-dtype'),
    pytest.param(lambda: sd.rec(8, out=np.zeros((8, 1), dtype='float64')),
                 id='rec-valid-out-float64'),
    pytest.param(lambda: sd.rec(8, mapping=[2, 1]),
                 id='rec-valid-mapping'),
    pytest.param(lambda: sd.rec(8, channels=1, dtype='uint8'),
                 id='rec-valid-uint8'),
]

# --- playrec() ---------------------------------------------------------

PLAYREC_INVALID = [
    pytest.param(lambda: sd.playrec(np.zeros((8, 1), dtype='complex64')),
                 sd.DtypeError, 'Unsupported data type',
                 id='playrec-dtype-data'),
    pytest.param(lambda: sd.playrec(np.zeros((8, 1)),
                                    out=np.zeros((8, 1), dtype='float16')),
                 sd.DtypeError, 'Unsupported data type',
                 id='playrec-dtype-out'),
    pytest.param(lambda: sd.playrec(np.zeros((2, 2, 2))),
                 sd.ShapeError, 'one- or two-dimensional',
                 id='playrec-shape-3d'),
    pytest.param(lambda: sd.playrec(np.zeros((16, 1)),
                                    out=np.zeros((8, 1))),
                 sd.ShapeError, r'len\(data\) != len\(out\)',
                 id='playrec-shape-length-mismatch'),
    pytest.param(lambda: sd.playrec(np.zeros((8, 2)),
                                    output_mapping=[1, 1]),
                 sd.MappingError, 'only appear once',
                 id='playrec-mapping-output-duplicate'),
    pytest.param(lambda: sd.playrec(np.zeros((8, 1)), channels=1,
                                    input_mapping=[1, 2]),
                 sd.MappingError, 'size of input mapping',
                 id='playrec-mapping-input-length-mismatch'),
]

PLAYREC_VALID = [
    pytest.param(lambda: sd.playrec(np.zeros((8, 1)), channels=1),
                 id='playrec-valid-mono'),
    pytest.param(lambda: sd.playrec(np.zeros((8, 2)), channels=2,
                                    input_mapping=[2, 1],
                                    output_mapping=[2, 1]),
                 id='playrec-valid-mappings'),
]

# --- Stream/InputStream/OutputStream ------------------------------------

STREAM_INVALID = [
    pytest.param(lambda: sd.InputStream(dtype='float64'),
                 sd.DtypeError, 'Invalid input dtype',
                 id='stream-dtype-float64'),
    pytest.param(lambda: sd.OutputStream(dtype='not-a-dtype'),
                 sd.DtypeError, 'Invalid output dtype',
                 id='stream-dtype-unknown'),
    pytest.param(lambda: sd.InputStream(dtype='int24'),
                 sd.DtypeError, "'int24' is not supported",
                 id='stream-dtype-int24-input'),
    pytest.param(lambda: sd.Stream(dtype='int24'),
                 sd.DtypeError, "'int24' is not supported",
                 id='stream-dtype-int24-duplex'),
    pytest.param(lambda: sd.InputStream(channels=0),
                 sd.ChannelsError, 'at least 1',
                 id='stream-channels-zero'),
    pytest.param(lambda: sd.OutputStream(channels=-2),
                 sd.ChannelsError, 'at least 1',
                 id='stream-channels-negative'),
    pytest.param(lambda: sd.Stream(channels=2.5),
                 sd.ChannelsError, 'must be an integer',
                 id='stream-channels-non-integer'),
    pytest.param(lambda: sd.InputStream(channels=3),  # fake device: max 2
                 sd.ChannelsError, 'at most 2',
                 id='stream-channels-exceeds-max'),
    pytest.param(lambda: sd.Stream(dtype=('float32', 'float64')),
                 sd.DtypeError, 'Invalid output dtype',
                 id='stream-duplex-dtype-pair'),
    pytest.param(lambda: sd.Stream(channels=(1, 0)),
                 sd.ChannelsError, 'at least 1',
                 id='stream-duplex-channels-pair'),
]

STREAM_VALID = [
    pytest.param(lambda: sd.Stream(channels=(1, 2), dtype='float32'),
                 id='stream-valid-duplex'),
    pytest.param(lambda: sd.InputStream(channels=1, dtype='int16'),
                 id='stream-valid-input-int16'),
    pytest.param(lambda: sd.OutputStream(channels=2, dtype='uint8'),
                 id='stream-valid-output-uint8'),
    pytest.param(lambda: sd.RawInputStream(channels=1, dtype='int24'),
                 id='stream-valid-raw-int24'),
]


@pytest.mark.parametrize('thunk,error,match',
                         PLAY_INVALID + REC_INVALID + PLAYREC_INVALID)
def test_convenience_invalid(fake_device, forbid_open_stream,
                             thunk, error, match):
    check_invalid(thunk, error, match)


@pytest.mark.parametrize('thunk,error,match', STREAM_INVALID)
def test_stream_invalid(fake_device, forbid_open_stream,
                        thunk, error, match):
    check_invalid(thunk, error, match)


@pytest.mark.parametrize('thunk', PLAY_VALID + REC_VALID + PLAYREC_VALID)
def test_convenience_valid(fake_device, thunk):
    check_valid(thunk)


@pytest.mark.parametrize('thunk', STREAM_VALID)
def test_stream_valid(fake_device, thunk):
    check_valid(thunk)


# --- taxonomy ------------------------------------------------------------

def test_exception_hierarchy():
    assert issubclass(sd.StreamValidationError, ValueError)
    for cls in sd.DtypeError, sd.ShapeError, sd.ChannelsError, sd.MappingError:
        assert issubclass(cls, sd.StreamValidationError)
        assert issubclass(cls, ValueError)  # backward compatible
    # dtype errors were historically raised as TypeError, too:
    assert issubclass(sd.DtypeError, TypeError)


def test_classified_errors_are_not_portaudio_errors():
    for cls in sd.DtypeError, sd.ShapeError, sd.ChannelsError, sd.MappingError:
        assert not issubclass(cls, sd.PortAudioError)


def test_float64_conversion_unchanged():
    # float64 is still converted to float32 for play()/rec()/playrec():
    assert sd._check_dtype('float64') == 'float32'
    assert sd._check_dtype(np.dtype('float64')) == 'float32'
    # ... but rejected for Stream:
    with pytest.raises(sd.DtypeError):
        sd._check_dtype('complex64')
