"""Tests for ROM data extraction and parameter ranges."""

import pytest
from pyvotrax.rom import ROM_DATA, PhonemeParams, _RAW_ROM


class TestROMData:
    def test_rom_has_64_entries(self):
        assert len(ROM_DATA) == 64

    def test_raw_rom_has_64_entries(self):
        assert len(_RAW_ROM) == 64

    def test_all_entries_are_phoneme_params(self):
        for i, p in enumerate(ROM_DATA):
            assert isinstance(p, PhonemeParams), f"Entry {i} is not PhonemeParams"

    def test_4bit_params_in_range(self):
        for i, p in enumerate(ROM_DATA):
            assert 0 <= p.f1 <= 15, f"f1 out of range at {i}: {p.f1}"
            assert 0 <= p.va <= 15, f"va out of range at {i}: {p.va}"
            assert 0 <= p.f2 <= 15, f"f2 out of range at {i}: {p.f2}"
            assert 0 <= p.fc <= 15, f"fc out of range at {i}: {p.fc}"
            assert 0 <= p.f2q <= 15, f"f2q out of range at {i}: {p.f2q}"
            assert 0 <= p.f3 <= 15, f"f3 out of range at {i}: {p.f3}"
            assert 0 <= p.fa <= 15, f"fa out of range at {i}: {p.fa}"
            assert 0 <= p.cld <= 15, f"cld out of range at {i}: {p.cld}"
            assert 0 <= p.vd <= 15, f"vd out of range at {i}: {p.vd}"

    def test_closure_is_0_or_1(self):
        for i, p in enumerate(ROM_DATA):
            assert p.closure in (0, 1), f"closure not 0/1 at {i}: {p.closure}"

    def test_duration_7bit(self):
        for i, p in enumerate(ROM_DATA):
            assert 0 <= p.duration <= 127, f"duration out of range at {i}: {p.duration}"

    def test_pause_phonemes(self):
        # PA0 (0x03) and PA1 (0x3E) are pause phonemes
        assert ROM_DATA[0x03].pause is False
        assert ROM_DATA[0x3E].pause is False

    def test_non_pause_phonemes(self):
        # Most phonemes are not pauses
        assert ROM_DATA[0x00].pause is True  # EH3
        assert ROM_DATA[0x24].pause is True  # AH
        assert ROM_DATA[0x3F].pause is True  # STOP

    def test_spot_check_eh3(self):
        """Spot check EH3 (code 0) against manually decoded values."""
        p = ROM_DATA[0]
        # word0=0x361 = 0b001101100001
        # Duration from word0 bits 5-11:
        #   bit5  (0x020): 0x361 & 0x020 = 0x020 -> 0x40
        #   bit6  (0x040): 0x361 & 0x040 = 0x040 -> 0x20
        #   bit7  (0x080): 0x361 & 0x080 = 0     -> 0
        #   bit8  (0x100): 0x361 & 0x100 = 0x100 -> 0x08
        #   bit9  (0x200): 0x361 & 0x200 = 0x200 -> 0x04
        #   bit10 (0x400): 0x361 & 0x400 = 0     -> 0
        #   bit11 (0x800): 0x361 & 0x800 = 0     -> 0
        # duration = 0x40 + 0x20 + 0x08 + 0x04 = 0x6C = 108
        assert p.duration == 108
        # closure: bit4 of word0: 0x361 & 0x10 = 0 -> 0
        assert p.closure == 0

    def test_nonzero_durations_exist(self):
        """At least some phonemes should have nonzero duration."""
        durations = [p.duration for p in ROM_DATA]
        assert max(durations) > 0
        # Most phonemes have duration > 0
        assert sum(1 for d in durations if d > 0) > 50
