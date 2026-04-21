"""Tests for the pure-Python SC-01A emulation (py_emu)."""

import numpy as np
import pytest
from py_emu.chip import VotraxSC01APython
from py_emu.rom import ROM_DATA


class TestInterpolation:
    def test_interpolation_converges(self):
        """Interpolation registers should converge toward target via formant interpolation."""
        chip = VotraxSC01APython()
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

        # reg >> 4 should be close to target
        assert abs((reg >> 4) - target) <= 1


class TestPitchCounter:
    def test_pitch_wraps_at_8bit(self):
        """Pitch counter should stay within 0-255."""
        chip = VotraxSC01APython()
        chip.phone_commit(0x24, 0)
        # Run many updates
        for _ in range(1000):
            chip._chip_update()
        assert 0 <= chip._pitch <= 255

    def test_pitch_reset_value(self):
        """Pitch should reset to 0 when reaching the pitch_reset value."""
        chip = VotraxSC01APython()
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
        chip = VotraxSC01APython()
        chip.phone_commit(0x24, 0)

        seen = set()
        for _ in range(100):
            chip._chip_update()
            seen.add(chip._noise)

        # Should have many distinct values
        assert len(seen) > 50, f"LFSR produced only {len(seen)} distinct values"

    def test_lfsr_stays_15bit(self):
        """LFSR should remain within 15-bit range."""
        chip = VotraxSC01APython()
        chip.phone_commit(0x24, 0)
        for _ in range(1000):
            chip._chip_update()
            assert 0 <= chip._noise <= 0x7FFF


class TestClosureAttenuation:
    def test_closure_range(self):
        """Closure should count up from 0 toward 28."""
        chip = VotraxSC01APython()
        chip.phone_commit(0x24, 0)  # AH
        # After some updates, closure should increase toward 28
        chip.generate_samples(2000)
        assert chip._closure >= 0
        assert chip._closure <= 28
