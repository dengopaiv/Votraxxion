"""Tests for the VotraxSC01A chip emulator."""

import numpy as np
import pytest
from pyvotrax.chip import VotraxSC01A
from pyvotrax.rom import ROM_DATA


class TestInterpolation:
    def test_interpolation_converges(self):
        """Interpolation registers should converge toward target * 16."""
        chip = VotraxSC01A()
        chip.phone_commit(0x24, 0)  # AH

        target = ROM_DATA[0x24].va
        # Run many interpolation steps
        for _ in range(200):
            chip._interpolate()

        # cur_va >> 4 should be close to target
        converged = chip._cur_va >> 4
        assert converged == target or abs(converged - target) <= 1, \
            f"Expected ~{target}, got {converged} (cur_va={chip._cur_va})"

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
        chip = VotraxSC01A()
        chip.phone_commit(0x24, 0)
        # Run many updates
        for _ in range(1000):
            chip._chip_update()
        assert 0 <= chip._pitch <= 255

    def test_pitch_reset_value(self):
        """Pitch should reset to 0 when reaching the pitch_reset value."""
        chip = VotraxSC01A()
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
        chip = VotraxSC01A()
        chip.phone_commit(0x24, 0)

        seen = set()
        for _ in range(100):
            chip._chip_update()
            seen.add(chip._noise)

        # Should have many distinct values
        assert len(seen) > 50, f"LFSR produced only {len(seen)} distinct values"

    def test_lfsr_stays_15bit(self):
        """LFSR should remain within 15-bit range."""
        chip = VotraxSC01A()
        chip.phone_commit(0x24, 0)
        for _ in range(1000):
            chip._chip_update()
            assert 0 <= chip._noise <= 0x7FFF


class TestDuration:
    def test_duration_formula(self):
        """Test the duration formula produces positive values."""
        chip = VotraxSC01A()
        chip.phone_commit(0x24, 0)  # AH
        dur = chip.get_phone_duration_samples()
        assert dur > 0

        # Verify against manual calculation
        rom_dur = ROM_DATA[0x24].duration
        expected_master = 16 * (rom_dur * 4 + 1) * 4 * 9 + 2
        expected_samples = expected_master // 18
        assert dur == expected_samples

    def test_all_durations_positive(self):
        """All phonemes should have positive duration."""
        chip = VotraxSC01A()
        for code in range(64):
            chip.phone_commit(code, 0)
            dur = chip.get_phone_duration_samples()
            assert dur > 0, f"Phoneme {code} has zero duration"


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
        """Closure should attenuate the start of a phoneme."""
        chip = VotraxSC01A()
        # Choose a phoneme with closure delay
        chip.phone_commit(0x24, 0)  # AH
        # closure should be initialized from ROM
        initial_closure = chip._closure
        # After some updates, closure should decrease
        chip.generate_samples(2000)
        # Closure counts down
        assert chip._closure <= initial_closure
