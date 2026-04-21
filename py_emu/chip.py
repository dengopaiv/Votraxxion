"""Votrax SC-01A chip-level emulation — pure Python DSP.

Emulates the SC-01A at the DSP level, faithfully reproducing the MAME
votrax.cpp analog signal path to generate speech waveforms.

Two rates:
- 40 kHz (SCLOCK): analog_calc() runs every sample
- 20 kHz (CCLOCK): chip_update() runs every other sample

This module also includes the enhanced mode DSP features (KLGLOTT88 glottal
model, PolyBLEP, nasal anti-resonators, per-fricative noise shaping, etc.)
for reference and experimentation. These are NOT part of the real SC-01A.
"""

import numpy as np

from .rom import ROM_DATA
from .filters import (
    SCLOCK, CCLOCK,
    bits_to_caps, build_standard_filter, build_noise_shaper_filter,
    build_lowpass_filter, build_injection_filter, apply_filter, shift_hist,
    build_enhanced_noise_filter,
)

# 9-level glottal waveform from MAME
_GLOTTAL = [0.0, -4.0/7.0, 1.0, 6.0/7.0, 5.0/7.0, 4.0/7.0, 3.0/7.0, 2.0/7.0, 1.0/7.0]

# Nasal phoneme codes and their anti-formant frequencies (Hz) / bandwidths (Hz)
_NASAL_ANTI_FORMANTS = {
    0x0C: (750.0, 200.0),   # M
    0x0D: (1450.0, 200.0),  # N
    0x14: (2500.0, 300.0),  # NG
}

# Fricative phoneme codes and their noise shaper parameters (center_freq, bandwidth)
_FRICATIVE_NOISE_PARAMS = {
    0x1F: (6800.0, 2000.0),  # S
    0x12: (6800.0, 2000.0),  # Z
    0x11: (3800.0, 1500.0),  # SH
    0x07: (3800.0, 1500.0),  # ZH
    0x1D: (7700.0, 4000.0),  # F
    0x0F: (7700.0, 4000.0),  # V
    0x39: (7500.0, 4000.0),  # TH
    0x38: (7500.0, 4000.0),  # THV
}


def _polyblep(t: float, dt: float) -> float:
    """PolyBLEP correction at discontinuities."""
    if t < dt:
        t /= dt
        return t + t - t * t - 1.0
    if t > 1.0 - dt:
        t = (t - 1.0) / dt
        return t * t + t + t + 1.0
    return 0.0


class VotraxSC01APython:
    """Votrax SC-01A speech synthesizer chip emulator — pure Python.

    This is the full Python DSP implementation including optional enhanced
    mode features. It exists as a development reference and fallback.

    Args:
        enhanced: Enable enhanced mode with KLGLOTT88 glottal source,
                  PolyBLEP anti-aliasing, jitter, and shimmer (default False).
    """

    def __init__(self, enhanced: bool = False, rd: float = 1.0):
        self._enhanced = enhanced
        self._rd = rd
        self.reset()

    def reset(self):
        """Power-on reset: initialize all state to defaults."""
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

        # Enhanced mode state
        self._radiation_prev = 0.0
        self._dc_prev_in = 0.0
        self._dc_prev_out = 0.0

        # LF model state
        self._lf_phase = 0.0
        self._lf_te = 0.0
        self._lf_tp = 0.0
        self._lf_ta = 0.0
        self._lf_E0 = 1.0
        self._lf_alpha = 0.0
        self._lf_omega_g = 0.0
        self._lf_Ee = 1.0
        self._lf_epsilon = 0.0
        self._lf_tc = 1.0
        self._lf_period_samples = 128.0

        # Nasal anti-resonator state
        self._nar_x = np.zeros(3)  # x[n], x[n-1], x[n-2]
        self._nar_active = False
        self._nar_A = 0.0
        self._nar_C = 0.0

        # Enhanced noise shaper coefficients (per-fricative)
        self._enh_ns_a = None
        self._enh_ns_b = None
        self._enh_ns_x = np.zeros(3)
        self._enh_ns_y = np.zeros(2)

        # Oversampling: 7-tap halfband FIR for decimation from 80kHz to 40kHz
        self._halfband_coeffs = np.array([
            -0.0234375, 0.0, 0.2734375, 0.5, 0.2734375, 0.0, -0.0234375
        ])
        self._oversample_buf = np.zeros(7)

        # Spectral tilt post-filter
        self._tilt_prev = 0.0

        # Build all filter coefficients
        self._build_fixed_filters()
        self._build_variable_filters()

        # Prime the LF glottal model with a default musical F0 so enhanced
        # mode has valid tp/alpha/E0/omega_g/period from sample 0. We
        # intentionally do NOT recompute these on phone_commit (that caused
        # mid-utterance clicks — see the note in phone_commit).
        if self._enhanced:
            self._compute_lf_params(120.0)

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
        self._phone = phone & 0x3F
        self._inflection = inflection & 0x03
        self._rom = ROM_DATA[self._phone]
        self._phonetick = 0
        self._ticks = 0
        # closure is set when ticks reaches cld, or immediately if cld==0
        if self._rom.cld == 0:
            self._cur_closure = self._rom.closure

        if self._enhanced:
            import math
            # LF glottal F0 is set once at reset() to a fixed musical value and
            # NOT recomputed here: the Votrax pitch formula depends on filt_f1,
            # so every phoneme change gave a different F0 and the LF
            # parameters (tp, alpha, E0, omega_g, period) all jumped
            # discontinuously while _lf_phase kept advancing — audible as a
            # click at each phoneme boundary. Keeping F0 constant across the
            # utterance trades away the authentic SC-01A F0-F1 coupling (which
            # the LF model doesn't reproduce anyway) for a click-free voice.

            # Nasal anti-resonator setup
            nasal_params = _NASAL_ANTI_FORMANTS.get(self._phone)
            if nasal_params is not None:
                freq, bw = nasal_params
                self._nar_active = True
                self._nar_C = -math.exp(-2.0 * math.pi * bw / SCLOCK)
                self._nar_A = 2.0 * math.exp(-math.pi * bw / SCLOCK) * math.cos(2.0 * math.pi * freq / SCLOCK)
                self._nar_x[:] = 0.0
            else:
                self._nar_active = False

            # Per-fricative noise shaping setup
            fric_params = _FRICATIVE_NOISE_PARAMS.get(self._phone)
            if fric_params is not None:
                center, bw = fric_params
                self._enh_ns_a, self._enh_ns_b = build_enhanced_noise_filter(center, bw)
                self._enh_ns_x[:] = 0.0
                self._enh_ns_y[:] = 0.0
            else:
                self._enh_ns_a = None
                self._enh_ns_b = None

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
        # Two-level duration counter (phonetick -> ticks)
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

    def _glottal_lf_half_step(self) -> float:
        """LF glottal model: generate one sample at 80 kHz (half phase step).

        Called twice per 40 kHz sample for 2x oversampling of the glottal source.
        """
        import math
        import random
        period = self._lf_period_samples if self._lf_period_samples > 0 else 128.0
        # dt for 80 kHz rate = half of 40 kHz step
        dt = 0.5 / period
        phase = self._lf_phase

        t0 = 1.0 / self._lf_f0 if self._lf_f0 > 0 else period / SCLOCK
        te_norm = self._lf_te / t0 if t0 > 0 else 0.55
        tc_norm = 1.0

        if phase < te_norm:
            t = phase * t0
            sample = self._lf_E0 * math.exp(self._lf_alpha * t) * math.sin(self._lf_omega_g * t)
        elif phase < tc_norm:
            t_rel = (phase - te_norm) * t0
            tc_rel = (tc_norm - te_norm) * t0
            eps = self._lf_epsilon
            if eps > 0 and self._lf_ta > 0:
                sample = (-self._lf_Ee / (eps * self._lf_ta)) * (
                    math.exp(-eps * t_rel) - math.exp(-eps * tc_rel)
                )
            else:
                sample = 0.0
        else:
            sample = 0.0

        # PolyBLEP at glottal closure
        closure_phase = phase - te_norm
        if closure_phase < 0.0:
            closure_phase += 1.0
        sample += _polyblep(closure_phase, dt) * 0.5

        # Shimmer (applied once per full sample, but small enough to be ok per half)
        sample *= 1.0 + random.gauss(0.0, 0.02)

        # Advance phase by half step with jitter
        jitter = 1.0 + random.gauss(0.0, 0.015)
        self._lf_phase += dt * jitter
        if self._lf_phase >= 1.0:
            self._lf_phase -= 1.0

        return sample

    def _compute_lf_params(self, f0: float):
        """Compute LF model parameters from Rd and F0 (once per phoneme)."""
        import math
        rd = self._rd
        self._lf_f0 = f0

        # Rd -> LF parameter mapping (Fant 1995)
        rap = max((-1.0 + 4.8 * rd) / 100.0, 0.01)
        rkp = (22.4 + 11.8 * rd) / 100.0
        rgp = rkp / (4.0 * rap * (0.5 + 1.2 * rkp))

        tp = 1.0 / (2.0 * f0 * rgp)
        te = tp * (1.0 + rkp)
        ta = rap / f0  # Ta in seconds
        t0 = 1.0 / f0

        self._lf_tp = tp
        self._lf_te = te
        self._lf_ta = ta
        self._lf_tc = t0

        # omega_g = pi / Tp
        self._lf_omega_g = math.pi / tp

        # epsilon = 1 / Ta
        self._lf_epsilon = 1.0 / ta if ta > 0 else 1e6

        # Solve for alpha using Newton-Raphson
        omega_g = self._lf_omega_g
        alpha = 0.0
        for _ in range(20):
            sin_te = math.sin(omega_g * te)
            cos_te = math.cos(omega_g * te)
            if abs(sin_te) < 1e-10:
                break
            exp_a_te = math.exp(alpha * te)
            f_val = omega_g * cos_te + alpha * sin_te
            f_deriv = sin_te
            if abs(f_deriv) < 1e-10:
                break
            alpha = alpha - f_val / f_deriv

        self._lf_alpha = alpha

        # E0: normalize so peak is ~1
        peak_val = math.exp(alpha * tp) * math.sin(omega_g * tp)
        self._lf_E0 = 1.0 / abs(peak_val) if abs(peak_val) > 1e-10 else 1.0

        # Ee: excitation amplitude at Te
        self._lf_Ee = abs(self._lf_E0 * math.exp(alpha * te) * math.sin(omega_g * te))

        # Period in samples
        self._lf_period_samples = SCLOCK / f0

    def _analog_calc(self) -> float:
        """40 kHz analog calculation: compute one output sample through the
        full filter cascade.

        Signal path:
        Glottal -> *va/15 -> F1 -> F2v --------+
                                                +--> F3 -> +noise -> F4 -> *closure -> FX -> *0.35
        Noise LFSR -> *fa/15 -> Shaper --+-----/             ^
                                         +-> *fc/15 -> F2n   |
                                         +-> *(5+(15^fc))/20-+

        Enhanced mode adds: radiation filter, DC blocker, LF glottal model,
        nasal anti-resonators, per-fricative noise shaping, oversampling,
        spectral tilt post-filter.
        """
        # --- Glottal source ---
        if self._enhanced:
            # 2x oversampling: generate 2 glottal samples at 80 kHz,
            # decimate through 7-tap halfband FIR
            g1 = self._glottal_lf_half_step()
            g2 = self._glottal_lf_half_step()

            # Push both through halfband FIR buffer and take output
            for s in (g1, g2):
                # Shift buffer right
                for k in range(6, 0, -1):
                    self._oversample_buf[k] = self._oversample_buf[k - 1]
                self._oversample_buf[0] = s

            # FIR output (decimated: take every 2nd)
            glottal = float(np.dot(self._halfband_coeffs, self._oversample_buf))
        else:
            glot_idx = self._pitch >> 3
            glottal = _GLOTTAL[glot_idx] if glot_idx < 9 else 0.0

        # --- Voice path: glottal * va/15 -> F1 -> F2v ---
        voice = glottal * (self._filt_va / 15.0)

        shift_hist(voice, self._f1_x)
        f1_out = apply_filter(self._f1_x, self._f1_y, self._f1_a, self._f1_b)
        shift_hist(f1_out, self._f1_y)

        # Enhanced: nasal anti-resonator between F1 and F2v
        if self._enhanced and self._nar_active:
            # y[n] = x[n] - A*x[n-1] - C*x[n-2]
            self._nar_x[2] = self._nar_x[1]
            self._nar_x[1] = self._nar_x[0]
            self._nar_x[0] = f1_out
            f1_out = (self._nar_x[0]
                      - self._nar_A * self._nar_x[1]
                      - self._nar_C * self._nar_x[2])

        shift_hist(f1_out, self._f2v_x)
        f2v_out = apply_filter(self._f2v_x, self._f2v_y, self._f2v_a, self._f2v_b)
        shift_hist(f2v_out, self._f2v_y)

        # --- Noise path: MAME scales noise by 1e4, gated by pitch bit 6 ---
        noise_gate = self._cur_noise if (self._pitch & 0x40) else False
        noise_raw = 1e4 * (1.0 if noise_gate else -1.0)

        # Enhanced: aspiration noise modulation by glottal phase
        if self._enhanced:
            te_norm = self._lf_te / self._lf_tc if self._lf_tc > 0 else 0.55
            noise_mod = 1.0 if self._lf_phase < te_norm else 0.3
            noise_in = noise_raw * (self._filt_fa / 15.0) * noise_mod
        else:
            noise_in = noise_raw * (self._filt_fa / 15.0)

        # Enhanced: per-fricative noise shaping
        if self._enhanced and self._enh_ns_a is not None:
            shift_hist(noise_in, self._enh_ns_x)
            ns_out = apply_filter(self._enh_ns_x, self._enh_ns_y,
                                  self._enh_ns_a, self._enh_ns_b)
            shift_hist(ns_out, self._enh_ns_y)
        else:
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

        if self._enhanced:
            # --- Radiation filter: +6 dB/oct first difference ---
            radiation_out = fx_out - self._radiation_prev
            self._radiation_prev = fx_out

            # --- DC blocking filter: 20 Hz cutoff ---
            import math
            R = 1.0 - (2.0 * math.pi * 20.0 / SCLOCK)
            dc_out = radiation_out - self._dc_prev_in + R * self._dc_prev_out
            self._dc_prev_in = radiation_out
            self._dc_prev_out = dc_out

            # --- Spectral tilt post-filter ---
            tilt_a = 0.0  # neutral by default
            tilt_out = (1.0 - tilt_a) * dc_out + tilt_a * self._tilt_prev
            self._tilt_prev = tilt_out

            return tilt_out * 0.35
        else:
            return fx_out * 0.35

    def generate_one_sample(self) -> float:
        """Generate a single audio sample at 40 kHz.

        chip_update() runs every other sample (20 kHz),
        analog_calc() runs every sample (40 kHz).
        """
        self._sample_count += 1
        if self._sample_count % 2 == 0:
            self._chip_update()
        return self._analog_calc()

    def generate_samples(self, n: int) -> np.ndarray:
        """Generate n audio samples at 40 kHz."""
        output = np.zeros(n)
        for i in range(n):
            output[i] = self.generate_one_sample()
        return output

    @property
    def phone_done(self) -> bool:
        """True when the current phoneme has finished (ticks reached 0x10)."""
        return self._ticks >= 0x10
