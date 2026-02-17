"""pyvotrax — Votrax SC-01A speech synthesizer DSP emulator.

Provides chip-level emulation of the Votrax SC-01A using numpy/scipy,
faithfully reproducing the MAME votrax.cpp analog signal path.
"""

from .chip import VotraxSC01A
from .synth import VotraxSynthesizer
from .phonemes import PHONE_TABLE, name_to_code, code_to_name

__all__ = [
    "VotraxSC01A",
    "VotraxSynthesizer",
    "PHONE_TABLE",
    "name_to_code",
    "code_to_name",
]
