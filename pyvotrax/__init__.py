"""pyvotrax — Votrax SC-01A speech synthesizer DSP emulator.

Provides chip-level emulation of the Votrax SC-01A using numpy/scipy,
faithfully reproducing the MAME votrax.cpp analog signal path.

When the C++ extension is compiled, the DSP core runs natively for
~20-50x faster sample generation. Use has_native_core() to check.
"""

from .chip import VotraxSC01A, _HAS_NATIVE
from .synth import VotraxSynthesizer
from .phonemes import PHONE_TABLE, name_to_code, code_to_name
from .tts import VotraxTTS


def has_native_core() -> bool:
    """Return True if the C++ DSP core extension is available."""
    return _HAS_NATIVE


__all__ = [
    "VotraxSC01A",
    "VotraxSynthesizer",
    "VotraxTTS",
    "PHONE_TABLE",
    "name_to_code",
    "code_to_name",
    "has_native_core",
]
