"""Tests for the text-to-speech pipeline."""

import os
import tempfile

import numpy as np
import pytest

from pyvotrax.phonemes import name_to_code
from pyvotrax.tts import (
    ARPABET_TO_VOTRAX,
    VotraxTTS,
    _VOWEL_VARIANTS,
    _select_variant,
    _strip_stress,
    _stress_to_inflection,
    _tokenize,
    _word_to_arpabet,
    arpabet_to_votrax,
)


class TestArpabetMapping:
    """Test ARPAbet → Votrax phoneme mapping coverage."""

    EXPECTED_ARPABET = [
        "AA", "AE", "AH", "AO", "AW", "AY",
        "B", "CH", "D", "DH",
        "EH", "ER", "EY",
        "F", "G", "HH",
        "IH", "IY",
        "JH", "K", "L", "M", "N", "NG",
        "OW", "OY",
        "P", "R", "S", "SH",
        "T", "TH",
        "UH", "UW",
        "V", "W", "Y", "Z", "ZH",
    ]

    def test_all_39_arpabet_phonemes_mapped(self):
        """All 39 ARPAbet phonemes have a Votrax mapping."""
        for phone in self.EXPECTED_ARPABET:
            assert phone in ARPABET_TO_VOTRAX, f"{phone} not in mapping"

    def test_mapping_count(self):
        assert len(ARPABET_TO_VOTRAX) == 39

    @pytest.mark.parametrize("arpa,votrax_names", [
        # AA (/ɑ/ "father") → Votrax AH (/ɑ/ "mop"); NOT Votrax A which is /eɪ/ "day".
        # AH (/ʌ/ "cup")    → Votrax UH (/ʌ/ "cup").
        ("AA", ["AH"]),
        ("AE", ["AE"]),
        ("AH", ["UH"]),
        ("AO", ["AW"]),
        ("AW", ["AW1"]),
        ("AY", ["AY"]),
        ("B", ["B"]),
        ("CH", ["CH"]),
        ("D", ["D"]),
        ("DH", ["THV"]),
        ("EH", ["EH"]),
        ("ER", ["ER"]),
        ("EY", ["E1"]),
        ("F", ["F"]),
        ("G", ["G"]),
        ("HH", ["H"]),
        ("IH", ["I"]),
        ("IY", ["I1"]),
        ("JH", ["J"]),
        ("K", ["K"]),
        ("L", ["L"]),
        ("M", ["M"]),
        ("N", ["N"]),
        ("NG", ["NG"]),
        ("OW", ["O1"]),
        ("OY", ["O1", "Y"]),
        ("P", ["P"]),
        ("R", ["R"]),
        ("S", ["S"]),
        ("SH", ["SH"]),
        ("T", ["T"]),
        ("TH", ["TH"]),
        ("UH", ["UH"]),
        ("UW", ["U1"]),
        ("V", ["V"]),
        ("W", ["W"]),
        ("Y", ["Y"]),
        ("Z", ["Z"]),
        ("ZH", ["ZH"]),
    ])
    def test_each_mapping(self, arpa, votrax_names):
        assert ARPABET_TO_VOTRAX[arpa] == votrax_names

    def test_all_votrax_names_are_valid(self):
        """Every Votrax name in the mapping resolves to a valid code."""
        for arpa, votrax_names in ARPABET_TO_VOTRAX.items():
            for vname in votrax_names:
                code = name_to_code(vname)
                assert 0 <= code <= 63, f"{arpa} -> {vname} -> {code} out of range"


class TestStressHandling:
    """Test stress digit stripping and inflection mapping."""

    def test_strip_stress_0(self):
        assert _strip_stress("AH0") == ("AH", 0)

    def test_strip_stress_1(self):
        assert _strip_stress("AH1") == ("AH", 1)

    def test_strip_stress_2(self):
        assert _strip_stress("AH2") == ("AH", 2)

    def test_strip_no_stress(self):
        assert _strip_stress("B") == ("B", -1)

    def test_inflection_primary(self):
        assert _stress_to_inflection(1) == 2

    def test_inflection_secondary(self):
        assert _stress_to_inflection(2) == 1

    def test_inflection_unstressed(self):
        assert _stress_to_inflection(0) == 0

    def test_inflection_consonant(self):
        assert _stress_to_inflection(-1) == 0


class TestArpabetToVotrax:
    """Test the arpabet_to_votrax conversion function."""

    def test_simple_vowel(self):
        # ARPAbet AH maps to Votrax UH (/ʌ/ "cup"); with no context and no next
        # phoneme, the variant selector picks the longest variant in the UH
        # group (["UH3", "UH2", "UH1", "UH"]) → "UH".
        result = arpabet_to_votrax(["AH0"])
        assert len(result) == 1
        assert result[0] == (name_to_code("UH"), 0)

    def test_stressed_vowel(self):
        result = arpabet_to_votrax(["AH1"])
        assert result[0] == (name_to_code("UH"), 2)

    def test_consonant(self):
        result = arpabet_to_votrax(["B"])
        assert result[0] == (name_to_code("B"), 0)

    def test_diphthong_split(self):
        """OY splits into O1 + Y."""
        result = arpabet_to_votrax(["OY1"])
        assert len(result) == 2
        assert result[0] == (name_to_code("O1"), 2)
        assert result[1] == (name_to_code("Y"), 2)

    def test_unknown_phone_skipped(self):
        result = arpabet_to_votrax(["XX"])
        assert result == []


class TestWordLookup:
    """Test CMU dictionary word lookup."""

    def test_hello(self):
        """'hello' should be found in CMU dict."""
        phones = _word_to_arpabet("hello")
        assert phones is not None
        assert len(phones) > 0

    def test_the(self):
        phones = _word_to_arpabet("the")
        assert phones is not None

    def test_unknown_word(self):
        phones = _word_to_arpabet("xyzzyplugh")
        assert phones is None


class TestTokenize:
    """Test text tokenization."""

    def test_simple(self):
        assert _tokenize("hello world") == ["hello", "world"]

    def test_punctuation(self):
        assert _tokenize("hello, world!") == ["hello", "world"]

    def test_apostrophe(self):
        assert _tokenize("don't") == ["don't"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_numbers_stripped(self):
        assert _tokenize("hello 123 world") == ["hello", "world"]


class TestVotraxTTS:
    """Integration tests for VotraxTTS."""

    @pytest.fixture
    def tts(self):
        return VotraxTTS()

    def test_text_to_phonemes_hello(self, tts):
        """'hello' produces a non-empty phoneme sequence ending with STOP."""
        phonemes = tts.text_to_phonemes("hello")
        assert len(phonemes) > 0
        # Last phoneme should be STOP
        assert phonemes[-1][0] == name_to_code("STOP")

    def test_text_to_phonemes_multi_word(self, tts):
        """Multi-word text includes PA0 pauses between words."""
        phonemes = tts.text_to_phonemes("hello world")
        pa0_code = name_to_code("PA0")
        # There should be at least one PA0 pause in the sequence
        codes = [p[0] for p in phonemes]
        assert pa0_code in codes

    def test_text_to_phonemes_empty(self, tts):
        assert tts.text_to_phonemes("") == []

    def test_text_to_phonemes_unknown_word(self, tts):
        """Unknown words produce phonemes via letter-by-letter fallback."""
        phonemes = tts.text_to_phonemes("xyzzy")
        assert len(phonemes) > 0
        assert phonemes[-1][0] == name_to_code("STOP")

    def test_speak_produces_audio(self, tts):
        audio = tts.speak("hello")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert audio.dtype == np.float64

    def test_speak_empty(self, tts):
        audio = tts.speak("")
        assert len(audio) == 0

    def test_speak_to_wav(self, tts):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            tts.speak_to_wav("hello", path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 44  # WAV header is 44 bytes
        finally:
            os.unlink(path)

    def test_speak_to_wav_custom_rate(self, tts):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            tts.speak_to_wav("hello", path, sample_rate=22050)
            from scipy.io import wavfile
            rate, data = wavfile.read(path)
            assert rate == 22050
        finally:
            os.unlink(path)


class TestVowelVariants:
    """Test the _VOWEL_VARIANTS data structure."""

    def test_all_variant_names_are_valid_phonemes(self):
        """Every variant name in _VOWEL_VARIANTS resolves to a valid code."""
        for base, variants in _VOWEL_VARIANTS.items():
            for vname in variants:
                code = name_to_code(vname)
                assert 0 <= code <= 63, f"{base} variant {vname} -> {code} out of range"

    def test_base_is_always_longest(self):
        """The base (undecorated) name is always the last (longest) variant."""
        for base, variants in _VOWEL_VARIANTS.items():
            assert variants[-1] == base, f"Expected {base} as last variant, got {variants[-1]}"

    def test_variant_groups_have_at_least_two(self):
        """Every variant group has at least 2 entries."""
        for base, variants in _VOWEL_VARIANTS.items():
            assert len(variants) >= 2, f"{base} has only {len(variants)} variants"

    def test_four_entry_groups(self):
        """EH, I, and UH have 4 variants."""
        for base in ("EH", "I", "UH"):
            assert len(_VOWEL_VARIANTS[base]) == 4

    def test_two_entry_groups(self):
        """OO, U, E, AE have 2 variants."""
        for base in ("OO", "U", "E", "AE"):
            assert len(_VOWEL_VARIANTS[base]) == 2


class TestSelectVariant:
    """Test the _select_variant function."""

    # --- Rule 1: Stress-based default ---

    def test_unstressed_gives_shortest(self):
        """Unstressed vowel → shortest variant."""
        assert _select_variant("EH", 0, "L", "N", False) == "EH3"

    def test_primary_stress_gives_longest(self):
        """Primary-stressed vowel → longest variant."""
        assert _select_variant("EH", 1, "L", "N", False) == "EH"

    def test_secondary_stress_gives_medium(self):
        """Secondary-stressed vowel → medium variant."""
        # EH has 4 variants: EH3, EH2, EH1, EH → n//2 = 2 → EH1
        assert _select_variant("EH", 2, "L", "N", False) == "EH1"

    def test_secondary_stress_three_entry_group(self):
        """Secondary stress with 3-entry group picks middle."""
        # A has 3 variants: A2, A1, A → n//2 = 1 → A1
        assert _select_variant("A", 2, "L", "N", False) == "A1"

    def test_secondary_stress_two_entry_group(self):
        """Secondary stress with 2-entry group picks first."""
        # E has 2 variants: E1, E → n//2 = 1 → E
        assert _select_variant("E", 2, "L", "N", False) == "E"

    # --- Rule 2: Consonant cluster override ---

    def test_between_stops_gives_shortest(self):
        """Vowel between two stops → shortest variant (tight context)."""
        assert _select_variant("EH", 1, "T", "K", False) == "EH3"

    def test_between_stop_and_affricate_gives_shortest(self):
        """Vowel between stop and affricate → shortest."""
        assert _select_variant("I", 1, "P", "CH", False) == "I3"

    def test_between_affricates_gives_shortest(self):
        """Vowel between two affricates → shortest."""
        assert _select_variant("AH", 1, "CH", "JH", False) == "AH2"

    def test_tight_context_overrides_stress(self):
        """Tight context overrides primary stress."""
        assert _select_variant("UH", 1, "D", "G", False) == "UH3"

    # --- Rule 2: Before pause → longest ---

    def test_before_pause_gives_longest(self):
        """Vowel before pause (next=None) → longest variant."""
        assert _select_variant("EH", 0, "L", None, True) == "EH"

    def test_before_pause_overrides_unstressed(self):
        """Even unstressed vowel before pause → longest."""
        assert _select_variant("I", 0, "L", None, True) == "I"

    # --- Rule 3: Word-final unstressed → shortest ---

    def test_word_final_unstressed_gives_shortest(self):
        """Word-final unstressed vowel → shortest (reduction)."""
        assert _select_variant("AH", 0, "L", "M", True) == "AH2"

    # --- Consonants/non-variant vowels pass through ---

    def test_consonant_passes_through(self):
        """Non-variant names are returned unchanged."""
        assert _select_variant("B", -1, None, None, False) == "B"

    def test_non_variant_vowel_passes_through(self):
        """Votrax names not in variant groups pass through."""
        assert _select_variant("AY", 1, None, None, False) == "AY"

    def test_er_passes_through(self):
        assert _select_variant("ER", 1, None, None, False) == "ER"


class TestCoarticulationIntegration:
    """Integration tests for coarticulation in arpabet_to_votrax."""

    def test_single_unstressed_vowel_before_pause(self):
        """Single unstressed vowel defaults to longest (before pause)."""
        # AH (/ʌ/) → Votrax UH; longest UH variant is "UH".
        result = arpabet_to_votrax(["AH0"])
        assert result[0] == (name_to_code("UH"), 0)

    def test_unstressed_vowel_in_context(self):
        """Unstressed vowel between consonants → shortest variant."""
        result = arpabet_to_votrax(["B", "EH0", "T"])
        # EH0: unstressed, prev=B, next=T → stress-based: shortest → EH3
        # (B and T are both stops, so tight context → EH3)
        eh_code = result[1][0]
        assert eh_code == name_to_code("EH3")

    def test_stressed_vowel_in_context(self):
        """Primary-stressed vowel between non-tight consonants → longest."""
        result = arpabet_to_votrax(["L", "EH1", "N"])
        eh_code = result[1][0]
        assert eh_code == name_to_code("EH")

    def test_better_different_variants(self):
        """'better' has unstressed and stressed vowels with different variants.

        CMU: B EH1 T ER0 → the EH1 between B(stop) and T(stop) should be
        shortest (tight context), while ER0 is not a variant vowel.
        """
        phones = ["B", "EH1", "T", "ER0"]
        result = arpabet_to_votrax(phones)
        # EH1 between B and T (both stops) → EH3 (tight context override)
        eh_code = result[1][0]
        assert eh_code == name_to_code("EH3")

    def test_last_word_vowel_before_pause(self):
        """Last vowel in last word gets longest variant (before sentence pause)."""
        result = arpabet_to_votrax(["M", "AH1"], is_last_word=True)
        # AH1 → Votrax UH; is_word_final=True, is_last_word=True → next=None → longest UH.
        uh_code = result[1][0]
        assert uh_code == name_to_code("UH")

    def test_ih_unstressed_between_consonants(self):
        """IH0 between non-tight consonants → shortest (stress-based)."""
        result = arpabet_to_votrax(["L", "IH0", "N"])
        i_code = result[1][0]
        assert i_code == name_to_code("I3")

    def test_consonant_unchanged(self):
        """Consonants are not affected by variant selection."""
        result = arpabet_to_votrax(["T"])
        assert result[0] == (name_to_code("T"), 0)

    def test_electronic_uses_ah_not_a_for_tron(self):
        """Regression: ARPAbet AA1 in 'electronic' (CMU: IH2 L EH2 K T R AA1 N IH0 K)
        must map to Votrax AH (/ɑ/ "mop"), not Votrax A (/eɪ/ "day"). Earlier
        versions produced 'electraynic'. User-reported regression."""
        from pyvotrax.tts import VotraxTTS
        from pyvotrax.phonemes import code_to_name

        tts = VotraxTTS()
        phonemes = tts.text_to_phonemes("electronic")
        names = [code_to_name(c) for c, _ in phonemes]
        # The /ɑ/ of "tron" should be an AH-family phoneme, not A-family.
        ah_family = {"AH", "AH1", "AH2"}
        a_family = {"A", "A1", "A2"}
        ah_count = sum(1 for n in names if n in ah_family)
        a_count = sum(1 for n in names if n in a_family)
        assert ah_count >= 1, (
            f"Expected at least one AH-family phoneme for /ɑ/ in 'electronic'; got {names}"
        )
        assert a_count == 0, (
            f"No A-family (/eɪ/) phonemes should appear in 'electronic'; got {names}"
        )

    def test_diphthong_oy_variant_applied(self):
        """OY diphthong: first element (O1) is not a variant base, passes through."""
        result = arpabet_to_votrax(["OY1"])
        assert result[0][0] == name_to_code("O1")
        assert result[1][0] == name_to_code("Y")
