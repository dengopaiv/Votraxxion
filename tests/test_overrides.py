"""Tests for per-phoneme parameter overrides."""

import numpy as np
import pytest

from pyvotrax.chip import VotraxSC01A, PHONEME_PARAM_FIELDS, rom_params
from pyvotrax.synth import VotraxSynthesizer
from pyvotrax.phonemes import name_to_code


class TestRomParams:
    def test_ah_has_all_fields(self):
        p = rom_params(name_to_code("AH"))
        assert set(p.keys()) == set(PHONEME_PARAM_FIELDS)
        assert isinstance(p["f1"], int)
        assert isinstance(p["pause"], bool)

    def test_pa0_is_pause(self):
        p = rom_params(name_to_code("PA0"))
        assert p["pause"] is True

    def test_ah_is_not_pause(self):
        p = rom_params(name_to_code("AH"))
        assert p["pause"] is False

    def test_out_of_range_is_masked(self):
        """Codes outside 0-63 should be masked (not raise)."""
        p = rom_params(0x7F)  # masked to 0x3F = STOP
        assert p == rom_params(0x3F)


class TestChipOverride:
    def test_noop_override_matches_rom(self):
        """phone_commit_override with no overrides should behave like phone_commit."""
        chip_a = VotraxSC01A()
        chip_a.phone_commit(name_to_code("AH"), 0)
        a = chip_a.generate_samples(4000)

        chip_b = VotraxSC01A()
        chip_b.phone_commit_override(name_to_code("AH"), 0)
        b = chip_b.generate_samples(4000)

        assert np.allclose(a, b)

    def test_f1_override_changes_output(self):
        """Overriding f1 on AH should produce audibly different output."""
        defaults = rom_params(name_to_code("AH"))

        chip_a = VotraxSC01A()
        chip_a.phone_commit(name_to_code("AH"), 0)
        a = chip_a.generate_samples(8000)

        chip_b = VotraxSC01A()
        # Push f1 to the opposite extreme from its ROM value
        new_f1 = 0 if defaults["f1"] > 7 else 15
        chip_b.phone_commit_override(name_to_code("AH"), 0, f1=new_f1)
        b = chip_b.generate_samples(8000)

        # Outputs should differ meaningfully
        diff = float(np.sqrt(np.mean((a - b) ** 2)))
        a_rms = float(np.sqrt(np.mean(a ** 2)))
        assert diff > 0.05 * a_rms, (
            f"f1 override produced near-identical output (diff={diff}, a_rms={a_rms})"
        )

    def test_unknown_override_raises(self):
        chip = VotraxSC01A()
        with pytest.raises(TypeError):
            chip.phone_commit_override(name_to_code("AH"), 0, bogus=1)

    def test_override_preserves_unspecified_fields(self):
        """Fields not in the override dict should come from ROM."""
        # Not directly inspectable without extra accessors; instead, verify
        # that specifying all-ROM fields explicitly gives the same output.
        ah = name_to_code("AH")
        defaults = rom_params(ah)

        chip_a = VotraxSC01A()
        chip_a.phone_commit(ah, 0)
        a = chip_a.generate_samples(4000)

        chip_b = VotraxSC01A()
        chip_b.phone_commit_override(ah, 0, **defaults)
        b = chip_b.generate_samples(4000)

        assert np.allclose(a, b)


class TestSynthesizerOverride:
    def test_3_tuple_accepts_override(self):
        synth = VotraxSynthesizer()
        ah = name_to_code("AH")
        audio = synth.synthesize([(ah, 0, {"f1": 0}), (name_to_code("STOP"), 0)])
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))

    def test_3_tuple_with_empty_dict_matches_2_tuple(self):
        synth_a = VotraxSynthesizer()
        a = synth_a.synthesize([(0x24, 0)])
        synth_b = VotraxSynthesizer()
        b = synth_b.synthesize([(0x24, 0, {})])
        assert np.allclose(a, b)

    def test_3_tuple_with_none_overrides(self):
        synth = VotraxSynthesizer()
        audio = synth.synthesize([(0x24, 0, None)])
        assert len(audio) > 0

    def test_invalid_tuple_length_raises(self):
        synth = VotraxSynthesizer()
        with pytest.raises(ValueError):
            synth.synthesize([(0x24,)])

    def test_override_produces_different_audio(self):
        """Same phoneme with different overrides should diverge."""
        synth_a = VotraxSynthesizer()
        synth_b = VotraxSynthesizer()
        ah = name_to_code("AH")
        a = synth_a.synthesize([(ah, 0)])
        b = synth_b.synthesize([(ah, 0, {"f1": 0, "f2": 0})])
        # Clip to the shorter length before comparing
        n = min(len(a), len(b))
        diff = float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))
        assert diff > 1e-3
