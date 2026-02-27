"""Votrax SC-01A chip-level emulation.

Emulates the SC-01A at the DSP level, faithfully reproducing the MAME
votrax.cpp analog signal path to generate speech waveforms.

Two rates:
- 40 kHz (SCLOCK): analog_calc() runs every sample
- 20 kHz (CCLOCK): chip_update() runs every other sample

When the C++ extension (_votrax_core) is available, VotraxSC01A delegates
to the native backend for ~20-50x faster sample generation. The pure-Python
implementation remains as a fallback.
"""

import numpy as np

from .rom import ROM_DATA
from .filters import (
    SCLOCK, CCLOCK,
    bits_to_caps, build_standard_filter, build_noise_shaper_filter,
    build_lowpass_filter, build_injection_filter, apply_filter, shift_hist,
)

# Try to import the C++ DSP core
try:
    from ._votrax_core import VotraxSC01ACore as _NativeCore
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False

# 9-level glottal waveform from MAME
_GLOTTAL = [0.0, -4.0/7.0, 1.0, 6.0/7.0, 5.0/7.0, 4.0/7.0, 3.0/7.0, 2.0/7.0, 1.0/7.0]


class VotraxSC01A:
    """Votrax SC-01A speech synthesizer chip emulator.

    When the C++ extension is available, delegates to VotraxSC01ACore for
    performance. Falls back to pure Python otherwise.

    Args:
        use_native: Use the C++ backend if available (default True).
        enhanced: Enable enhanced mode with KLGLOTT88 glottal source,
                  PolyBLEP anti-aliasing, jitter, and shimmer (default False).
    """

    def __init__(self, use_native: bool = True, enhanced: bool = False):
        self._native = None
        self._enhanced = enhanced
        if use_native and _HAS_NATIVE:
            self._native = _NativeCore(enhanced)
        self.reset()

    def reset(self):
        """Power-on reset: initialize all state to defaults."""
        if self._native is not None:
            self._native.reset()
            return
        # Phoneme state
        self._phone = 0x3F  # STOP
        self._inflection = 0
        self._rom = ROM_DATA[0x3F]

        # Interpolation registers (8-bit)
        self._cur_fa = 0
        self._cur_fc = 0
        self._cur_va = 0
        self._cur_f1 = 0
        self._cur_f2 = 0
        self._cur_f2q = 0
        self._cur_f3 = 0

        # Committed filter values (4-bit, except f2 which is 5-bit)
        self._filt_fa = 0
        self._filt_fc = 0
        self._filt_va = 0
        self._filt_f1 = 0
        self._filt_f2 = 0   # 5-bit
        self._filt_f2q = 0
        self._filt_f3 = 0

        # Timing
        self._phonetick = 0
        self._ticks = 0
        self._pitch = 0
        self._closure = 0
        self._cur_closure = True
        self._update_counter = 0
        self._sample_count = 0

        # Noise LFSR (15-bit)
        self._noise = 0
        self._cur_noise = False

        # Filter histories
        # Standard filters (4 a-coeff, 4 b-coeff): x[4], y[3]
        self._f1_x = np.zeros(4)
        self._f1_y = np.zeros(3)
        self._f2v_x = np.zeros(4)
        self._f2v_y = np.zeros(3)
        self._f3_x = np.zeros(4)
        self._f3_y = np.zeros(3)
        self._f4_x = np.zeros(4)
        self._f4_y = np.zeros(3)

        # Noise shaper (3 a-coeff, 3 b-coeff): x[3], y[2]
        self._ns_x = np.zeros(3)
        self._ns_y = np.zeros(2)

        # Injection filter F2n (2 a-coeff, 2 b-coeff): x[2], y[1]
        self._f2n_x = np.zeros(2)
        self._f2n_y = np.zeros(1)

        # Final lowpass FX (2 a-coeff, 2 b-coeff): x[2], y[1]
        self._fx_x = np.zeros(2)
        self._fx_y = np.zeros(1)

        # Build all filter coefficients
        self._build_fixed_filters()
        self._build_variable_filters()

    def _build_fixed_filters(self):
        """Build filters with fixed component values (F4, FX, noise shaper)."""
        # F4: fixed bandpass
        self._f4_a, self._f4_b = build_standard_filter(
            c1t=0, c1b=28810, c2t=1165, c2b=21457, c3=8558, c4=7289
        )
        # FX: final lowpass
        self._fx_a, self._fx_b = build_lowpass_filter(c1t=1122, c1b=23131)
        # Noise shaper: fixed
        self._ns_a, self._ns_b = build_noise_shaper_filter(
            c1=15500, c2t=14854, c2b=8450, c3=9523, c4=14083
        )

    def _build_variable_filters(self):
        """Build filters whose coefficients depend on current phoneme params."""
        # F1: c3 varies with filt_f1
        f1_c3 = 2280 + bits_to_caps(self._filt_f1, (2546, 4973, 9861, 19724))
        self._f1_a, self._f1_b = build_standard_filter(
            c1t=11247, c1b=11797, c2t=949, c2b=52067, c3=f1_c3, c4=166272
        )

        # F2v: c2t varies with filt_f2q, c3 varies with filt_f2 (5-bit)
        f2v_c2t = 829 + bits_to_caps(self._filt_f2q, (1390, 2965, 5875, 11297))
        f2v_c3 = 2352 + bits_to_caps(self._filt_f2, (833, 1663, 3164, 6327, 12654))
        self._f2v_a, self._f2v_b = build_standard_filter(
            c1t=24840, c1b=29154, c2t=f2v_c2t, c2b=38180, c3=f2v_c3, c4=34270
        )

        # F3: c3 varies with filt_f3
        f3_c3 = 8480 + bits_to_caps(self._filt_f3, (2226, 4485, 9056, 18111))
        self._f3_a, self._f3_b = build_standard_filter(
            c1t=0, c1b=17594, c2t=868, c2b=18828, c3=f3_c3, c4=50019
        )

        # F2n: injection filter (neutralized per MAME)
        self._f2n_a, self._f2n_b = build_injection_filter(
            c1b=29154,
            c2t=829 + bits_to_caps(self._filt_f2q, (1390, 2965, 5875, 11297)),
            c2b=38180,
            c3=2352 + bits_to_caps(self._filt_f2, (833, 1663, 3164, 6327, 12654)),
            c4=34270
        )

    def phone_commit(self, phone: int, inflection: int = 0):
        """Latch a new phoneme and begin generating it.

        Args:
            phone: 6-bit phoneme code (0-63)
            inflection: 2-bit inflection value (0-3)
        """
        if self._native is not None:
            self._native.phone_commit(phone, inflection)
            return
        self._phone = phone & 0x3F
        self._inflection = inflection & 0x03
        self._rom = ROM_DATA[self._phone]
        self._phonetick = 0
        self._ticks = 0
        # closure is set when ticks reaches cld, or immediately if cld==0
        if self._rom.cld == 0:
            self._cur_closure = self._rom.closure

    def _interpolate_formants(self):
        """Interpolate formant registers toward ROM targets at ~417 Hz.

        Formula: reg = (reg - (reg >> 3) + (target << 1)) & 0xFF
        Only formant parameters (fc, f1, f2, f2q, f3) — NOT amplitudes (fa, va).
        """
        rom = self._rom
        self._cur_fc = (self._cur_fc - (self._cur_fc >> 3) + (rom.fc << 1)) & 0xFF
        self._cur_f1 = (self._cur_f1 - (self._cur_f1 >> 3) + (rom.f1 << 1)) & 0xFF
        self._cur_f2 = (self._cur_f2 - (self._cur_f2 >> 3) + (rom.f2 << 1)) & 0xFF
        self._cur_f2q = (self._cur_f2q - (self._cur_f2q >> 3) + (rom.f2q << 1)) & 0xFF
        self._cur_f3 = (self._cur_f3 - (self._cur_f3 >> 3) + (rom.f3 << 1)) & 0xFF

    def _commit_filters(self):
        """Commit interpolated values to filter parameters and rebuild filters."""
        self._filt_f1 = self._cur_f1 >> 4
        self._filt_va = self._cur_va >> 4
        self._filt_f2 = self._cur_f2 >> 3   # 5-bit (key detail)
        self._filt_fc = self._cur_fc >> 4
        self._filt_f2q = self._cur_f2q >> 4
        self._filt_f3 = self._cur_f3 >> 4
        self._filt_fa = self._cur_fa >> 4
        self._build_variable_filters()

    def _chip_update(self):
        """20 kHz chip update: timing, interpolation, pitch, noise LFSR.

        Matches MAME's two-level timing system with separate interpolation
        rates for formants (~417 Hz) and amplitudes (~1250 Hz).
        """
        # Two-level duration counter (phonetick → ticks)
        if self._ticks != 0x10:
            self._phonetick += 1
            if self._phonetick == ((self._rom.duration << 2) | 1):
                self._phonetick = 0
                self._ticks += 1
                if self._ticks == self._rom.cld:
                    self._cur_closure = self._rom.closure

        # Update counter: period 48
        self._update_counter = (self._update_counter + 1) % 0x30

        tick_625 = not (self._update_counter & 0xF)
        tick_208 = self._update_counter == 0x28

        # Formant interpolation at ~417 Hz, with pause gating
        if tick_208 and (not self._rom.pause or not (self._filt_fa or self._filt_va)):
            self._interpolate_formants()

        # Amplitude interpolation at ~1250 Hz, with delay gating
        if tick_625:
            if self._ticks >= self._rom.vd:
                self._cur_fa = (self._cur_fa - (self._cur_fa >> 3) + (self._rom.fa << 1)) & 0xFF
            if self._ticks >= self._rom.cld:
                self._cur_va = (self._cur_va - (self._cur_va >> 3) + (self._rom.va << 1)) & 0xFF

        # Closure: counts UP from 0 to 28
        if not self._cur_closure and (self._filt_fa or self._filt_va):
            self._closure = 0
        elif self._closure != (7 << 2):
            self._closure += 1

        # Pitch counter (== not >=)
        self._pitch = (self._pitch + 1) & 0xFF
        if self._pitch == ((0xE0 ^ (self._inflection << 5) ^ (self._filt_f1 << 1)) + 2):
            self._pitch = 0

        # Filter commit
        if (self._pitch & 0xF9) == 0x08:
            self._commit_filters()

        # Noise LFSR (MAME feedback)
        inp = 1 if (self._cur_noise and self._noise != 0x7FFF) else 0
        self._noise = ((self._noise << 1) & 0x7FFE) | inp
        self._cur_noise = not (((self._noise >> 14) ^ (self._noise >> 13)) & 1)

    def _analog_calc(self) -> float:
        """40 kHz analog calculation: compute one output sample through the
        full filter cascade.

        Signal path:
        Glottal -> *va/15 -> F1 -> F2v --------+
                                                +--> F3 -> +noise -> F4 -> *closure -> FX -> *0.35
        Noise LFSR -> *fa/15 -> Shaper --+-----/             ^
                                         +-> *fc/15 -> F2n   |
                                         +-> *(5+(15^fc))/20-+
        """
        # --- Glottal source ---
        glot_idx = self._pitch >> 3
        glottal = _GLOTTAL[glot_idx] if glot_idx < 9 else 0.0

        # --- Voice path: glottal * va/15 -> F1 -> F2v ---
        voice = glottal * (self._filt_va / 15.0)

        shift_hist(voice, self._f1_x)
        f1_out = apply_filter(self._f1_x, self._f1_y, self._f1_a, self._f1_b)
        shift_hist(f1_out, self._f1_y)

        shift_hist(f1_out, self._f2v_x)
        f2v_out = apply_filter(self._f2v_x, self._f2v_y, self._f2v_a, self._f2v_b)
        shift_hist(f2v_out, self._f2v_y)

        # --- Noise path: MAME scales noise by 1e4, gated by pitch bit 6 ---
        noise_gate = self._cur_noise if (self._pitch & 0x40) else False
        noise_raw = 1e4 * (1.0 if noise_gate else -1.0)
        noise_in = noise_raw * (self._filt_fa / 15.0)

        shift_hist(noise_in, self._ns_x)
        ns_out = apply_filter(self._ns_x, self._ns_y, self._ns_a, self._ns_b)
        shift_hist(ns_out, self._ns_y)

        # Noise through F2n (FIR highpass approximation)
        noise_f2n_in = ns_out * (self._filt_fc / 15.0)
        shift_hist(noise_f2n_in, self._f2n_x)
        f2n_out = apply_filter(self._f2n_x, self._f2n_y, self._f2n_a, self._f2n_b)
        shift_hist(f2n_out, self._f2n_y)

        # Noise direct injection: shaped_noise * (5 + (15 ^ fc)) / 20
        noise_direct = ns_out * (5.0 + (15 ^ self._filt_fc)) / 20.0

        # --- Combine voice and noise -> F3 ---
        combined = f2v_out + f2n_out

        shift_hist(combined, self._f3_x)
        f3_out = apply_filter(self._f3_x, self._f3_y, self._f3_a, self._f3_b)
        shift_hist(f3_out, self._f3_y)

        # Add noise injection after F3
        f3_plus_noise = f3_out + noise_direct

        # --- F4 (fixed bandpass) ---
        shift_hist(f3_plus_noise, self._f4_x)
        f4_out = apply_filter(self._f4_x, self._f4_y, self._f4_a, self._f4_b)
        shift_hist(f4_out, self._f4_y)

        # --- Closure attenuation ---
        closure_atten = (7 ^ (self._closure >> 2)) / 7.0
        closure_out = f4_out * closure_atten

        # --- Final lowpass (FX) ---
        shift_hist(closure_out, self._fx_x)
        fx_out = apply_filter(self._fx_x, self._fx_y, self._fx_a, self._fx_b)
        shift_hist(fx_out, self._fx_y)

        return fx_out * 0.35

    def generate_one_sample(self) -> float:
        """Generate a single audio sample at 40 kHz.

        chip_update() runs every other sample (20 kHz),
        analog_calc() runs every sample (40 kHz).
        """
        if self._native is not None:
            return self._native.generate_one_sample()
        self._sample_count += 1
        if self._sample_count % 2 == 0:
            self._chip_update()
        return self._analog_calc()

    def generate_samples(self, n: int) -> np.ndarray:
        """Generate n audio samples at 40 kHz."""
        if self._native is not None:
            return self._native.generate_samples(n)
        output = np.zeros(n)
        for i in range(n):
            output[i] = self.generate_one_sample()
        return output

    @property
    def phone_done(self) -> bool:
        """True when the current phoneme has finished (ticks reached 0x10)."""
        if self._native is not None:
            return self._native.phone_done
        return self._ticks >= 0x10
