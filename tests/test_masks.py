"""The two SC-01 mask revisions.

The 1980 SC-01 and the later SC-01-A differ in exactly twelve ROM rows, and
every difference is the voice amplitude of an open vowel. These tests pin that
down on both backends, so a future edit to either ROM table has to be
deliberate.
"""

import numpy as np
import pytest

from py_emu.rom import MaskRevision as PyMask, rom_table
from pyvotrax.chip import (
    PHONEME_PARAM_FIELDS, MaskRevision, VotraxSC01A, rom_params,
)
from pyvotrax.phonemes import PHONE_TABLE, name_to_code

#: phone -> (SC-01-A va, 1980 SC-01 va). Decoded from the two mask ROM dumps.
EXPECTED_VA = {
    0x08: (9, 15),    # AH2
    0x13: (11, 15),   # AW1
    0x15: (9, 15),    # AH1
    0x23: (14, 15),   # UH3
    0x24: (9, 15),    # AH
    0x2E: (11, 15),   # AE
    0x2F: (11, 15),   # AE1
    0x30: (11, 15),   # AW2
    0x31: (14, 15),   # UH2
    0x32: (14, 15),   # UH1
    0x33: (14, 15),   # UH
    0x3D: (11, 15),   # AW
}


class TestMaskDifferences:
    def test_exactly_twelve_phonemes_differ(self):
        differing = [i for i in range(64)
                     if rom_params(i, MaskRevision.SC01A)
                     != rom_params(i, MaskRevision.SC01)]
        assert differing == sorted(EXPECTED_VA)

    def test_only_va_differs(self):
        """No field other than voice amplitude changed between the masks."""
        for i in range(64):
            a = rom_params(i, MaskRevision.SC01A)
            b = rom_params(i, MaskRevision.SC01)
            for field in PHONEME_PARAM_FIELDS:
                if field != "va":
                    assert a[field] == b[field], (i, field)

    @pytest.mark.parametrize("phone", sorted(EXPECTED_VA))
    def test_va_values(self, phone):
        want_a, want_sc01 = EXPECTED_VA[phone]
        assert rom_params(phone, MaskRevision.SC01A)["va"] == want_a
        assert rom_params(phone, MaskRevision.SC01)["va"] == want_sc01

    def test_all_sc01_vowels_are_full_scale(self):
        """The 1980 part ran every one of the twelve at va=15."""
        for phone in EXPECTED_VA:
            assert rom_params(phone, MaskRevision.SC01)["va"] == 15

    def test_differing_phonemes_are_all_open_vowels(self):
        names = {PHONE_TABLE[i] for i in EXPECTED_VA}
        assert names == {"AH2", "AW1", "AH1", "UH3", "AH", "AE",
                         "AE1", "AW2", "UH2", "UH1", "UH", "AW"}


class TestBackendsAgree:
    @pytest.mark.parametrize("cpp,py", [
        (MaskRevision.SC01A, PyMask.SC01A),
        (MaskRevision.SC01, PyMask.SC01),
    ])
    def test_cpp_matches_pure_python(self, cpp, py):
        for i in range(64):
            a = rom_params(i, cpp)
            b = rom_table(py)[i]
            for field in PHONEME_PARAM_FIELDS:
                assert a[field] == getattr(b, field), (i, field)


class TestMaskSelection:
    def test_default_is_sc01a(self):
        assert VotraxSC01A().mask == MaskRevision.SC01A
        assert rom_params(0x24)["va"] == 9

    def test_constructor_selects_mask(self):
        assert VotraxSC01A(mask=MaskRevision.SC01).mask == MaskRevision.SC01

    def test_mask_is_settable(self):
        chip = VotraxSC01A()
        chip.mask = MaskRevision.SC01
        assert chip.mask == MaskRevision.SC01

    def test_override_baseline_follows_the_mask(self):
        """phone_commit_override starts from the active mask's parameters."""
        chip = VotraxSC01A(mask=MaskRevision.SC01)
        chip.reset()
        # Overriding only f1 must leave va at the 1980 value, not the -A one.
        chip.phone_commit_override(name_to_code("AH"), 0, f1=7)
        assert rom_params(name_to_code("AH"), chip.mask)["va"] == 15


class TestAudibleDifference:
    def _render(self, mask):
        chip = VotraxSC01A(mask=mask)
        chip.reset()
        out = []
        for name in ("H", "AH", "L", "OO", "PA1", "AE", "N", "D"):
            chip.phone_commit(name_to_code(name), 0)
            guard = 0
            while not chip.phone_done and guard < 40000:
                out.append(chip.generate_one_sample())
                guard += 1
        return np.array(out)

    def test_sc01_is_louder(self):
        a = self._render(MaskRevision.SC01A)
        b = self._render(MaskRevision.SC01)
        assert len(a) == len(b)          # timing is identical; only level moves
        assert not np.array_equal(a, b)
        rms_a = np.sqrt((a ** 2).mean())
        rms_b = np.sqrt((b ** 2).mean())
        assert rms_b > rms_a * 1.2
