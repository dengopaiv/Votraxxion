"""Backward-compatible shim: re-exports clock constants and filter functions.

Clock constants are canonical in pyvotrax.constants.
Filter builder functions live in py_emu.filters.
"""

from .constants import MASTER_CLOCK, SCLOCK, CCLOCK

from py_emu.filters import (
    bits_to_caps,
    build_standard_filter,
    build_noise_shaper_filter,
    build_lowpass_filter,
    build_injection_filter,
    build_enhanced_noise_filter,
    apply_filter,
    shift_hist,
)

__all__ = [
    "MASTER_CLOCK", "SCLOCK", "CCLOCK",
    "bits_to_caps",
    "build_standard_filter",
    "build_noise_shaper_filter",
    "build_lowpass_filter",
    "build_injection_filter",
    "build_enhanced_noise_filter",
    "apply_filter",
    "shift_hist",
]
