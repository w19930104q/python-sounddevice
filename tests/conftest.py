"""Test configuration.

Imports the *sounddevice* module from ``../src`` (not any installed
copy) and makes sure a PortAudio library can be found.

If PortAudio is not installed system-wide, set the environment
variable ``PORTAUDIO_LIBRARY`` to the full path of the shared library,
e.g.::

    PORTAUDIO_LIBRARY=/path/to/libportaudio.so.2 python -m pytest tests/

The tests themselves do not require any audio hardware: all invalid
combinations are rejected before a stream is opened, and device
queries are stubbed where necessary.
"""

import ctypes.util
import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   os.pardir, 'src')
sys.path.insert(0, os.path.abspath(SRC))

_library = (os.environ.get('PORTAUDIO_LIBRARY') or
            ctypes.util.find_library('portaudio'))

if _library is None:
    pytest.skip('PortAudio library not found; set the PORTAUDIO_LIBRARY '
                'environment variable to its location',
                allow_module_level=True)

_original_find_library = ctypes.util.find_library


def _find_library(name):
    if name == 'portaudio':
        return _library
    return _original_find_library(name)


# sounddevice.py does "from ctypes.util import find_library" at import
# time, so patching it here (before the import) is sufficient:
ctypes.util.find_library = _find_library
