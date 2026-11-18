Validation and Error Classification Matrix
==========================================

.. currentmodule:: sounddevice

Before a stream is opened (and therefore before PortAudio can report
its own error codes), the combination of *dtype*, array *shape*,
number of *channels* and channel *mapping* is validated.
Invalid combinations raise one of the classified errors below --
all of them subclasses of `StreamValidationError` (and thereby of
`ValueError`) -- instead of a `PortAudioError` carrying an opaque
``PaErrorCode``:

.. autosummary::
   :nosignatures:

   StreamValidationError
   DtypeError
   ShapeError
   ChannelsError
   MappingError

The Matrix
----------

Rows are the entry points, columns are the validated aspects.
Each cell names the error raised for an invalid combination.
"---" means the aspect does not apply to that entry point.

.. list-table::
   :header-rows: 1
   :widths: 12 22 22 22 22

   * -
     - **dtype**
     - **shape**
     - **channels**
     - **mapping**
   * - `play()`
     - Unsupported *data* dtype → `DtypeError`
       (`float64` data is converted to `float32`)
     - *data* with more than two dimensions → `ShapeError`
     - (derived from *data* and *mapping*)
     - Channel number < 1, non-integer or empty mapping,
       duplicate channel numbers, mapping length !=
       columns of *data* → `MappingError`
   * - `rec()`
     - Unsupported *dtype* or dtype of *out* → `DtypeError`
       (`float64` is recorded as `float32` and converted afterwards)
     - *out* is not two-dimensional, negative *frames* → `ShapeError`;
       non-integer *frames* → `TypeError`
     - *channels* < 1 → `ChannelsError`;
       non-integer *channels* → `TypeError`
     - Channel number < 1, non-integer or empty mapping,
       mapping length != columns of *out* → `MappingError`
   * - `playrec()`
     - Like `play()` (output) and `rec()` (input) → `DtypeError`
     - Like `play()`/`rec()`, plus ``len(data) != len(out)``
       → `ShapeError`
     - Like `rec()`
     - *input_mapping*/*output_mapping*, otherwise like
       `play()`/`rec()` → `MappingError`
   * - `Stream`, `InputStream`, `OutputStream`
     - Unsupported *dtype*, `float64`, or `int24` on a NumPy stream
       → `DtypeError` (`int24` stays valid on `RawStream` and friends)
     - ---
     - *channels* < 1, non-integer *channels*, or *channels*
       exceeding the device's maximum → `ChannelsError`
     - ---

Notes
-----

* All classified errors are raised *before* the stream is opened,
  i.e. before `PortAudioError` could be raised by PortAudio itself.
* `DtypeError` is additionally a subclass of `TypeError`, because
  historically both `TypeError` and `ValueError` were raised for
  invalid *dtype* arguments.
* Valid combinations behave exactly as before: `float64` data is
  still converted from/to `float32` in `play()`/`rec()`/`playrec()`,
  mono data is still duplicated to all mapped channels, `int24` is
  still accepted by the "raw" stream classes, and all previously
  valid dtype/shape/channels/mapping combinations open streams
  unchanged.
* The matrix is verified by the acceptance tests in
  ``tests/test_validation_matrix.py`` (each test ID encodes one
  matrix cell).
