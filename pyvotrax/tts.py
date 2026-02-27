"""Text-to-speech pipeline using CMU Pronouncing Dictionary.

Converts English text to Votrax SC-01A phoneme sequences via ARPAbet mapping,
then synthesizes audio using the chip emulator.
"""

import re

import numpy as np
import cmudict

from .phonemes import name_to_code

_CMU_DICT = None


def _get_cmu_dict():
    global _CMU_DICT
    if _CMU_DICT is None:
        _CMU_DICT = cmudict.dict()
    return _CMU_DICT
from .synth import VotraxSynthesizer

# Vowel variant groups: shortest → longest duration.
# Numbered variants share identical formant targets but differ in closure delay,
# voice delay, and duration — encoding coarticulation context.
_VOWEL_VARIANTS = {
    "EH": ["EH3", "EH2", "EH1", "EH"],
    "I":  ["I3",  "I2",  "I1",  "I"],
    "A":  ["A2",  "A1",  "A"],
    "AH": ["AH2", "AH1", "AH"],
    "O":  ["O2",  "O1",  "O"],
    "UH": ["UH3", "UH2", "UH1", "UH"],
    "OO": ["OO1", "OO"],
    "AW": ["AW2", "AW1", "AW"],
    "U":  ["U1",  "U"],
    "E":  ["E1",  "E"],
    "AE": ["AE1", "AE"],
}

# Consonant classes for context rules.
_STOPS = {"P", "B", "T", "D", "K", "G"}
_AFFRICATES = {"CH", "JH"}
_TIGHT_CONTEXT = _STOPS | _AFFRICATES  # → shortest variant


def _select_variant(
    base_votrax: str,
    stress: int,
    prev_arpabet: str | None,
    next_arpabet: str | None,
    is_word_final: bool,
) -> str:
    """Select the context-appropriate vowel variant.

    Args:
        base_votrax: Base Votrax name from ARPABET_TO_VOTRAX (e.g., "EH").
        stress: ARPAbet stress level (0=unstressed, 1=primary, 2=secondary, -1=consonant).
        prev_arpabet: Previous ARPAbet phoneme (base, no stress digit), or None.
        next_arpabet: Next ARPAbet phoneme (base, no stress digit), or None.
        is_word_final: Whether this vowel is the last phoneme in the word.

    Returns:
        The selected Votrax variant name (e.g., "EH2").
    """
    variants = _VOWEL_VARIANTS.get(base_votrax)
    if variants is None:
        return base_votrax  # Not a variant group (consonant or non-variant vowel)

    n = len(variants)  # 2, 3, or 4 entries

    # Rule 2 override: vowel between two stops/affricates → shortest
    prev_tight = prev_arpabet in _TIGHT_CONTEXT if prev_arpabet else False
    next_tight = next_arpabet in _TIGHT_CONTEXT if next_arpabet else False
    if prev_tight and next_tight:
        return variants[0]

    # Rule 2 override: vowel before pause/word boundary → longest
    if next_arpabet is None:
        return variants[-1]

    # Rule 3: word-final unstressed vowel → shortest (reduction)
    if is_word_final and stress == 0:
        return variants[0]

    # Rule 1: stress-based default
    if stress == 0:
        # Unstressed → shortest
        return variants[0]
    elif stress == 2:
        # Secondary stress → medium
        return variants[n // 2]
    else:
        # Primary stress (1) or consonant (-1) → longest
        return variants[-1]


# ARPAbet → Votrax phoneme mapping.
# Stress digits (0/1/2) are stripped from vowels before lookup.
# Some ARPAbet phonemes map to multiple Votrax phonemes (e.g., OY → O1 + Y).
ARPABET_TO_VOTRAX = {
    "AA": ["A"],
    "AE": ["AE"],
    "AH": ["AH"],
    "AO": ["AW"],
    "AW": ["AW1"],
    "AY": ["AY"],
    "B":  ["B"],
    "CH": ["CH"],
    "D":  ["D"],
    "DH": ["THV"],
    "EH": ["EH"],
    "ER": ["ER"],
    "EY": ["E1"],
    "F":  ["F"],
    "G":  ["G"],
    "HH": ["H"],
    "IH": ["I"],
    "IY": ["I1"],
    "JH": ["J"],
    "K":  ["K"],
    "L":  ["L"],
    "M":  ["M"],
    "N":  ["N"],
    "NG": ["NG"],
    "OW": ["O1"],
    "OY": ["O1", "Y"],
    "P":  ["P"],
    "R":  ["R"],
    "S":  ["S"],
    "SH": ["SH"],
    "T":  ["T"],
    "TH": ["TH"],
    "UH": ["UH"],
    "UW": ["U1"],
    "V":  ["V"],
    "W":  ["W"],
    "Y":  ["Y"],
    "Z":  ["Z"],
    "ZH": ["ZH"],
}

# Letter-by-letter fallback for words not in CMU dict.
# Maps each letter to a rough ARPAbet-like pronunciation.
_LETTER_FALLBACK = {
    "A": ["EY"],
    "B": ["B", "IY"],
    "C": ["S", "IY"],
    "D": ["D", "IY"],
    "E": ["IY"],
    "F": ["EH", "F"],
    "G": ["JH", "IY"],
    "H": ["EY", "CH"],
    "I": ["AY"],
    "J": ["JH", "EY"],
    "K": ["K", "EY"],
    "L": ["EH", "L"],
    "M": ["EH", "M"],
    "N": ["EH", "N"],
    "O": ["OW"],
    "P": ["P", "IY"],
    "Q": ["K", "Y", "UW"],
    "R": ["AA", "R"],
    "S": ["EH", "S"],
    "T": ["T", "IY"],
    "U": ["Y", "UW"],
    "V": ["V", "IY"],
    "W": ["D", "AH", "B", "AH", "L", "Y", "UW"],
    "X": ["EH", "K", "S"],
    "Y": ["W", "AY"],
    "Z": ["Z", "IY"],
}


def _strip_stress(arpabet_phone: str) -> tuple[str, int]:
    """Strip stress digit from an ARPAbet phoneme.

    Returns (phoneme, stress) where stress is 0, 1, or 2.
    Consonants return stress=-1 (no stress).
    """
    if arpabet_phone and arpabet_phone[-1] in "012":
        return arpabet_phone[:-1], int(arpabet_phone[-1])
    return arpabet_phone, -1


def _stress_to_inflection(stress: int) -> int:
    """Map ARPAbet stress to Votrax inflection using all 4 levels.

    The SC-01A supports 4 inflection levels (0-3) that modulate pitch via
    inflection << 5 in the pitch counter. Using all levels produces more
    natural prosody than the original binary mapping.

    Primary stress (1) → inflection 2 (highest normal pitch)
    Secondary stress (2) → inflection 1 (moderate pitch)
    Unstressed (0) or consonant (-1) → inflection 0 (lowest pitch)
    """
    if stress == 1:
        return 2
    if stress == 2:
        return 1
    return 0


def arpabet_to_votrax(
    arpabet_phones: list[str],
    is_last_word: bool = False,
) -> list[tuple[int, int]]:
    """Convert a list of ARPAbet phonemes to Votrax (code, inflection) tuples.

    Performs context-dependent vowel variant selection: vowels in tight
    consonant clusters get shorter variants, stressed vowels get longer ones.

    Args:
        arpabet_phones: List of ARPAbet phoneme strings (e.g., ["HH", "AH0", "L", "OW1"])
        is_last_word: If True, the last phoneme is treated as word-final before a pause.

    Returns:
        List of (votrax_code, inflection) tuples.
    """
    # Pre-parse all phones to (base, stress) pairs for context lookups.
    parsed = [_strip_stress(phone) for phone in arpabet_phones]

    result = []
    for i, (base, stress) in enumerate(parsed):
        inflection = _stress_to_inflection(stress)

        votrax_names = ARPABET_TO_VOTRAX.get(base)
        if votrax_names is None:
            continue

        # Determine surrounding context for variant selection.
        prev_base = parsed[i - 1][0] if i > 0 else None
        next_base = parsed[i + 1][0] if i < len(parsed) - 1 else None
        is_word_final = (i == len(parsed) - 1)

        # If this is the last word and the vowel is word-final,
        # next_arpabet is None (before pause). Otherwise pass next phone.
        next_for_variant = None if (is_word_final and is_last_word) else next_base

        for vname in votrax_names:
            # Apply variant selection to the first Votrax name in the mapping.
            # (Diphthongs like OY→["O1","Y"] only vary the first element.)
            if vname == votrax_names[0]:
                vname = _select_variant(vname, stress, prev_base, next_for_variant, is_word_final)
            code = name_to_code(vname)
            result.append((code, inflection))

    return result


def _word_to_arpabet(word: str) -> list[str] | None:
    """Look up a word in the CMU dictionary.

    Returns the first pronunciation as a list of ARPAbet phonemes, or None.
    """
    phones = _get_cmu_dict().get(word.lower())
    if phones:
        return phones[0]
    return None


def _spell_word(word: str) -> list[tuple[int, int]]:
    """Letter-by-letter fallback for unknown words."""
    result = []
    for ch in word.upper():
        arpabet = _LETTER_FALLBACK.get(ch)
        if arpabet:
            result.extend(arpabet_to_votrax(arpabet))
            # Small pause between letters
            result.append((name_to_code("PA0"), 0))
    return result


def _tokenize(text: str) -> list[str]:
    """Split text into words, stripping punctuation."""
    return re.findall(r"[a-zA-Z']+", text)


def _detect_sentence_type(text: str) -> str:
    """Detect sentence type from trailing punctuation.

    Returns 'question' for ?, 'exclamation' for !, 'statement' otherwise.
    """
    stripped = text.rstrip()
    if stripped.endswith("?"):
        return "question"
    if stripped.endswith("!"):
        return "exclamation"
    return "statement"


class VotraxTTS:
    """Text-to-speech engine using CMU dict and the Votrax SC-01A emulator."""

    def __init__(self, enhanced: bool = False):
        self._synth = VotraxSynthesizer(enhanced=enhanced)

    def text_to_phonemes(self, text: str) -> list[tuple[int, int]]:
        """Convert text to a list of (votrax_code, inflection) tuples.

        Words are looked up in the CMU Pronouncing Dictionary.
        Unknown words are spelled out letter-by-letter.
        PA0 pauses are inserted between words, and STOP at the end.

        Sentence-level prosody:
        - Questions get rising inflection (level 3) on final stressed syllable
        - Statements get falling inflection (level 0) on final phonemes
        - Pre-pausal lengthening: extra PA0 before final pause
        """
        words = _tokenize(text)
        if not words:
            return []

        sentence_type = _detect_sentence_type(text)

        result = []
        pa0_code = name_to_code("PA0")
        stop_code = name_to_code("STOP")

        for i, word in enumerate(words):
            is_last = (i == len(words) - 1)
            arpabet = _word_to_arpabet(word)
            if arpabet is not None:
                result.extend(arpabet_to_votrax(arpabet, is_last_word=is_last))
            else:
                result.extend(_spell_word(word))

            # Insert pause between words
            if i < len(words) - 1:
                result.append((pa0_code, 0))

        # Apply sentence-final contour
        if result and sentence_type == "question":
            # Rising inflection: set the last few phonemes to inflection 3
            for j in range(len(result) - 1, max(len(result) - 4, -1), -1):
                code, inf = result[j]
                if code != pa0_code and code != stop_code:
                    result[j] = (code, 3)
        elif result and sentence_type == "statement":
            # Falling inflection: set the last few phonemes to inflection 0
            for j in range(len(result) - 1, max(len(result) - 3, -1), -1):
                code, inf = result[j]
                if code != pa0_code and code != stop_code:
                    result[j] = (code, 0)

        # Pre-pausal lengthening: add extra PA0 before STOP
        result.append((pa0_code, 0))
        result.append((stop_code, 0))
        return result

    def speak(self, text: str) -> np.ndarray:
        """Convert text to audio (40 kHz float64 array)."""
        phonemes = self.text_to_phonemes(text)
        if not phonemes:
            return np.array([], dtype=np.float64)
        return self._synth.synthesize(phonemes)

    def speak_to_wav(self, text: str, filename: str, sample_rate: int = 44100):
        """Convert text to a WAV file.

        Args:
            text: English text to synthesize.
            filename: Output WAV file path.
            sample_rate: Target sample rate (default 44100 Hz).
        """
        audio = self.speak(text)
        VotraxSynthesizer.to_wav(audio, filename, target_rate=sample_rate)
