"""Tests for the VotraxSC01A chip emulator."""

import numpy as np
import pytest
from pyvotrax.chip import VotraxSC01A
from pyvotrax.filters import SCLOCK
from pyvotrax.rom import ROM_DATA


class TestInterpolation:
    def test_interpolation_converges(self):
        """Interpolation registers should converge toward target via formant interpolation."""
        chip = VotraxSC01A(use_native=False)
        chip.phone_commit(0x24, 0)  # AH

        target = ROM_DATA[0x24].f1
        # Run many interpolation steps
        for _ in range(200):
            chip._interpolate_formants()

        # cur_f1 >> 4 should be close to target
        converged = chip._cur_f1 >> 4
        assert converged == target or abs(converged - target) <= 1, \
            f"Expected ~{target}, got {converged} (cur_f1={chip._cur_f1})"

    def test_interpolation_formula(self):
        """Test the interpolation formula directly."""
        # reg = (reg - (reg >> 3) + (target << 1)) & 0xFF
        reg = 0
        target = 10
        for _ in range(100):
            reg = (reg - (reg >> 3) + (target << 1)) & 0xFF

        # Should converge to target * 16 / 7 ≈ target * 2.28
        # Actually reg converges to target << 1 * 8/7 ≈ target * 2.28 * 8 ≈ 18.3
        # In practice, reg >> 4 should be close to target
        assert abs((reg >> 4) - target) <= 1


class TestPitchCounter:
    def test_pitch_wraps_at_8bit(self):
        """Pitch counter should stay within 0-255."""
        chip = VotraxSC01A(use_native=False)
        chip.phone_commit(0x24, 0)
        # Run many updates
        for _ in range(1000):
            chip._chip_update()
        assert 0 <= chip._pitch <= 255

    def test_pitch_reset_value(self):
        """Pitch should reset to 0 when reaching the pitch_reset value."""
        chip = VotraxSC01A(use_native=False)
        inflection = 1
        chip._inflection = inflection
        chip._filt_f1 = 5
        pitch_reset = ((0xE0 ^ (inflection << 5) ^ (5 << 1)) + 2) & 0xFF
        # Set pitch to just below reset point
        chip._pitch = pitch_reset - 1
        chip._chip_update()
        # Should have wrapped to 0
        assert chip._pitch == 0, f"Expected 0, got {chip._pitch} (reset={pitch_reset})"


class TestNoiseLFSR:
    def test_lfsr_nondegenerate(self):
        """LFSR should not get stuck at 0 or all-ones."""
        chip = VotraxSC01A(use_native=False)
        chip.phone_commit(0x24, 0)

        seen = set()
        for _ in range(100):
            chip._chip_update()
            seen.add(chip._noise)

        # Should have many distinct values
        assert len(seen) > 50, f"LFSR produced only {len(seen)} distinct values"

    def test_lfsr_stays_15bit(self):
        """LFSR should remain within 15-bit range."""
        chip = VotraxSC01A(use_native=False)
        chip.phone_commit(0x24, 0)
        for _ in range(1000):
            chip._chip_update()
            assert 0 <= chip._noise <= 0x7FFF


class TestDuration:
    def test_phone_done_eventually(self):
        """Phoneme should complete (phone_done becomes True)."""
        chip = VotraxSC01A()
        chip.phone_commit(0x24, 0)  # AH
        assert not chip.phone_done
        # Generate enough samples to complete
        for _ in range(200000):
            chip.generate_one_sample()
            if chip.phone_done:
                break
        assert chip.phone_done, "Phoneme did not complete"

    def test_all_phonemes_complete(self):
        """All phonemes should eventually complete."""
        chip = VotraxSC01A()
        for code in range(64):
            chip.phone_commit(code, 0)
            for _ in range(400000):
                chip.generate_one_sample()
                if chip.phone_done:
                    break
            assert chip.phone_done, f"Phoneme {code} did not complete"


class TestGenerateSamples:
    def test_output_shape(self):
        """generate_samples should return the requested number of samples."""
        chip = VotraxSC01A()
        chip.phone_commit(0x24, 0)  # AH (voiced)
        samples = chip.generate_samples(1000)
        assert samples.shape == (1000,)

    def test_output_finite(self):
        """All output samples should be finite."""
        chip = VotraxSC01A()
        chip.phone_commit(0x24, 0)  # AH (voiced)
        samples = chip.generate_samples(2000)
        assert np.all(np.isfinite(samples))

    def test_voiced_phoneme_not_silent(self):
        """A voiced phoneme (AH) should produce non-zero output."""
        chip = VotraxSC01A()
        chip.phone_commit(0x24, 0)  # AH
        samples = chip.generate_samples(4000)
        # Allow some initial silence, but overall should have energy
        rms = np.sqrt(np.mean(samples ** 2))
        assert rms > 1e-10, f"Output is silent (RMS={rms})"

    def test_closure_attenuation(self):
        """Closure should count up from 0 toward 28."""
        chip = VotraxSC01A(use_native=False)
        chip.phone_commit(0x24, 0)  # AH
        # After some updates, closure should increase toward 28
        chip.generate_samples(2000)
        assert chip._closure >= 0
        assert chip._closure <= 28


def _spectral_centroid(phoneme_code, n_samples=16000):
    """Compute the power-weighted spectral centroid above 1 kHz for a phoneme.

    Skips the first 4000 samples (transient) and uses the next 8000 (steady-state).
    Uses power spectrum (squared magnitudes) and excludes frequencies below 1 kHz
    to avoid DC/low-freq artifacts distorting the centroid.
    """
    chip = VotraxSC01A()
    chip.phone_commit(phoneme_code, 0)
    samples = chip.generate_samples(n_samples)
    steady = samples[4000:12000]
    power = np.abs(np.fft.rfft(steady)) ** 2
    freqs = np.fft.rfftfreq(len(steady), d=1.0 / SCLOCK)
    mask = freqs >= 1000
    total = np.sum(power[mask])
    if total == 0:
        return 0.0
    return np.sum(freqs[mask] * power[mask]) / total


class TestSibilantDifferentiation:
    # Phoneme codes: S=0x1F, SH=0x11, Z=0x12, ZH=0x07, CH=0x10, J=0x1A
    SIBILANTS = {"S": 0x1F, "SH": 0x11, "Z": 0x12, "ZH": 0x07, "CH": 0x10, "J": 0x1A}

    def test_sh_and_s_spectrally_different(self):
        """S (fc=0, direct path) and SH (fc=15, F2N path) should differ by >500 Hz."""
        centroid_s = _spectral_centroid(self.SIBILANTS["S"])
        centroid_sh = _spectral_centroid(self.SIBILANTS["SH"])
        diff = abs(centroid_s - centroid_sh)
        assert diff > 500, (
            f"S centroid={centroid_s:.0f} Hz, SH centroid={centroid_sh:.0f} Hz, "
            f"diff={diff:.0f} Hz — expected >500 Hz"
        )

    def test_sibilant_output_finite(self):
        """All sibilant phonemes should produce finite output (no NaN/Inf)."""
        for name, code in self.SIBILANTS.items():
            chip = VotraxSC01A()
            chip.phone_commit(code, 0)
            samples = chip.generate_samples(4000)
            assert np.all(np.isfinite(samples)), f"{name} (0x{code:02X}) has non-finite samples"
