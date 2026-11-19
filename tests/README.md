# Long-run stability regression for python-sounddevice

These tests guard against the class of bug that only shows up after a
process has opened and closed audio streams for a long time: a file
descriptor or thread that is not released on close, or resident memory
that creeps upward session after session. They never modify the
sounddevice implementation — they only exercise its public API.

## What is asserted

Two properties, over many real PortAudio stream lifecycles in one
process:

1. **Resources return to the starting point.** After a warm-up session a
   baseline of resident memory (RSS), open file descriptors and thread
   count is taken. Every scenario then creates and destroys streams
   repeatedly. File descriptors and threads must return **exactly** to
   baseline (a single leaked PulseAudio socket fails the run); RSS must
   stay within a fixed tolerance (`SD_LONGRUN_RSS_TOL_KB`, default
   2048 kB) and must not show a monotonic upward trend across
   checkpoints.
2. **Samples match a one-shot session.** A deterministic continuous-phase
   two-tone signal is played into a PulseAudio sink and recorded back from
   its `.monitor` source (a real in-kernel loopback, no cable or
   microphone). Startup priming blocks are discarded so only steady-state
   audio is compared. Each repeated session is checked against the single
   baseline session in two layers:
   * **Core invariants (hard gate):** the dominant frequency of each
     channel is the expected tone, spectral energy is concentrated at that
     tone, the level is healthy and non-silent, and no block is repeated
     (a stall). These survive the timing jitter a virtualized audio
     backend introduces while still catching corruption, a wrong signal,
     a dead/silent channel or a duplicated block.
   * **Strict similarity (deterministic backends):** best-lag
     cross-correlation and normalized spectral deviation against the
     reference; byte identity is reported when present. On a jittery
     backend a strict-layer miss with core invariants intact is reported
     as host jitter, not a failure.

The **normal** teardown path (`stop()` + `close()`, including the `with`
context manager) and the **abnormal-interrupt** path (`abort()` +
`close()`, exceptions raised while a stream is active, rejected stream
opens) are covered as separate tests. A subprocess test additionally
checks that `Pa_Initialize`/`Pa_Terminate` stay balanced.

## Running

The helper script points the loader at the bundled PortAudio library
(Linux; the shipped object has SONAME `libportaudio.so.2`) and runs the
suite isolated mode:

```sh
./tests/run_regression.sh              # quick gate (~1 minute)
./tests/run_regression.sh --soak       # extended profile
SD_LONGRUN_CYCLES=200000 \
    ./tests/run_regression.sh --soak   # multi-day soak
```

Equivalent plain pytest invocation (the `LD_LIBRARY_PATH` setup is only
needed when no system `libportaudio` is installed):

```sh
pytest tests/ -v
pytest tests/ -m longrun --soak-cycles 500
```

### Knobs (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `SD_LONGRUN_CYCLES` | 60 (2000 with `--soak`) | normal create/destroy cycles |
| `SD_LONGRUN_ABORT_CYCLES` | 30 (1000 with `--soak`) | `abort()`-path cycles |
| `SD_LONGRUN_BLOCKS` | 20 | audio blocks captured per session |
| `SD_LONGRUN_RSS_TOL_KB` | 2048 | max accepted RSS growth, kB |

## Requirements / behaviour on unsupported hosts

* A PulseAudio backend exposing a sink and its `.monitor` source. The
  audio tests are **skipped** (not failed) when no loopback pair exists,
  so the suite stays green on headless CI without audio.
* Resource accounting reads `/proc`, i.e. the exact FD/thread assertions
  apply on Linux.
* Some PortAudio host backends (notably PulseAudio under WSLg) ignore the
  `paComplete`/`paAbort` callback return values, so the tests always end
  streams with explicit `stop()`/`abort()` + `close()` rather than
  relying on `CallbackStop`. This is a property of the backend, not of
  sounddevice.

## Files

* `test_longrun_stability.py` — the test cases.
* `regression_support.py` — `/proc` resource accounting, loopback session
  driver and shift-tolerant content comparison.
* `conftest.py` — markers, `--soak`/`--soak-cycles` options and the
  environment-driven cycle/tolerance configuration.
* `run_regression.sh` — reproducible entry point.
