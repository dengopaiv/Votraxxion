"""py_emu — Pure Python Votrax SC-01A DSP emulation.

This package contains the full Python DSP implementation of the Votrax SC-01A
chip emulator, including the enhanced mode features (KLGLOTT88, nasal
anti-resonators, per-fricative noise shaping, etc.). It exists as a
development reference and fallback for when the C++ extension is not available.

The production DSP path is the C++ backend in pyvotrax._votrax_core.
"""

from .chip import VotraxSC01APython

__all__ = ["VotraxSC01APython"]
