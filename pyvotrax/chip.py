"""Votrax SC-01A chip-level emulation — C++ backend wrapper.

Delegates all DSP work to the compiled C++ extension (_votrax_core).
For the pure-Python implementation, see py_emu.VotraxSC01APython.
"""

import numpy as np

from .constants import MASTER_CLOCK as _DEFAULT_MASTER_CLOCK

try:
    from ._votrax_core import VotraxSC01ACore as _NativeCore
    from ._votrax_core import PhonemeParams as _NativeParams
except ImportError:
    raise ImportError(
        "The C++ DSP extension (_votrax_core) is not compiled. "
        "Build it with: python setup.py build_ext --inplace\n"
        "For the pure-Python fallback, use: from py_emu import VotraxSC01APython"
    )

# The 12 fields of a PhonemeParams struct, in the constructor's positional order.
PHONEME_PARAM_FIELDS = (
    "f1", "va", "f2", "fc", "f2q", "f3", "fa",
    "cld", "vd", "closure", "duration", "pause",
)


def rom_params(phone: int) -> dict:
    """Return the ROM-decoded phoneme parameters for ``phone`` (0-63) as a dict.

    Keys match :data:`PHONEME_PARAM_FIELDS`. Useful for UIs that need to show
    defaults next to user overrides.
    """
    p = _NativeCore.rom_params(int(phone) & 0x3F)
    return {f: getattr(p, f) for f in PHONEME_PARAM_FIELDS}


class VotraxSC01A:
    """Votrax SC-01A speech synthesizer chip emulator.

    Wraps the C++ VotraxSC01ACore for high-performance sample generation.

    Args:
        master_clock: Master clock frequency in Hz. Nominal 720 000.
            The 1980 datasheet endorses varying this for sound-design effects.
            Smaller values slow down and lower the pitch; larger speed up and
            raise pitch. Internally SCLOCK = master_clock/18, CCLOCK = /36.
        fx_fudge: Final-stage lowpass cutoff scaling. ``150/4000`` (default)
            matches MAME's observed chip behavior (~4 kHz authentic cutoff).
            Pass ``1.0`` for the muffled "as-schematic" 150 Hz behavior.
    """

    def __init__(
        self,
        master_clock: float = _DEFAULT_MASTER_CLOCK,
        fx_fudge: float = 150.0 / 4000.0,
        closure_strength: float = 1.0,
    ):
        self._master_clock = float(master_clock)
        self._fx_fudge = float(fx_fudge)
        self._closure_strength = float(closure_strength)
        self._native = _NativeCore(
            self._master_clock, self._fx_fudge, self._closure_strength
        )

    @property
    def master_clock(self) -> float:
        return self._master_clock

    @property
    def sclock(self) -> float:
        """Analog sample rate in Hz (master_clock / 18)."""
        return self._master_clock / 18.0

    @property
    def cclock(self) -> float:
        """Chip update rate in Hz (master_clock / 36)."""
        return self._master_clock / 36.0

    @property
    def fx_fudge(self) -> float:
        return self._fx_fudge

    @property
    def closure_strength(self) -> float:
        return self._closure_strength

    def reset(self):
        """Power-on reset: initialize all state to defaults."""
        self._native.reset()

    def phone_commit(self, phone: int, inflection: int = 0):
        """Latch a new phoneme and begin generating it.

        Args:
            phone: 6-bit phoneme code (0-63)
            inflection: 2-bit inflection value (0-3)
        """
        self._native.phone_commit(phone, inflection)

    def phone_commit_override(
        self,
        phone: int,
        inflection: int = 0,
        **overrides,
    ):
        """Latch a phoneme with per-field parameter overrides.

        Starts from the ROM-decoded parameters for ``phone`` and replaces any
        field whose name appears in ``overrides``. Recognized override keys:
        ``f1``, ``va``, ``f2``, ``fc``, ``f2q``, ``f3``, ``fa``, ``cld``,
        ``vd``, ``closure``, ``duration``, ``pause``. Field meanings are
        documented on :class:`pyvotrax._votrax_core.PhonemeParams`.

        Raises TypeError on unknown override keys, to catch typos early.
        """
        unknown = set(overrides) - set(PHONEME_PARAM_FIELDS)
        if unknown:
            raise TypeError(
                f"Unknown PhonemeParams override fields: {sorted(unknown)!r}"
            )
        base = _NativeCore.rom_params(int(phone) & 0x3F)
        merged = {f: getattr(base, f) for f in PHONEME_PARAM_FIELDS}
        merged.update(overrides)
        params = _NativeParams(**merged)
        self._native.phone_commit_override(int(phone) & 0x3F, int(inflection) & 0x03, params)

    def generate_one_sample(self) -> float:
        """Generate a single audio sample at ``self.sclock``."""
        return self._native.generate_one_sample()

    def generate_samples(self, n: int) -> np.ndarray:
        """Generate n audio samples at ``self.sclock``."""
        return self._native.generate_samples(n)

    @property
    def phone_done(self) -> bool:
        """True when the current phoneme has finished (ticks reached 0x10)."""
        return self._native.phone_done
