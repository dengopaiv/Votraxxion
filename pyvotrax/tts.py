"""Text-to-speech pipeline using CMU Pronouncing Dictionary.

Converts English text to Votrax SC-01A phoneme sequences via ARPAbet mapping,
then synthesizes audio using the chip emulator.
"""

import math
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

# ARPAbet vowels (for detecting voiced/voiceless context)
_ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}

# Voiceless consonants in ARPAbet (for Klatt rule 6)
_VOICELESS = {"P", "T", "K", "F", "TH", "S", "SH", "CH", "HH"}

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
    # Vowel mapping note: ARPAbet AA (/ɑ/ as in "father", "tron") and AH
    # (/ʌ/ as in "cup") correspond to Votrax AH and UH respectively per the
    # SC-01A datasheet phoneme chart (AH="mop", UH="cup"). The Votrax "A"
    # phoneme is /eɪ/ ("day") and must not be used for ARPAbet AA — earlier
    # versions had AA→A and AH→AH, which mispronounced "electronic" as
    # "electraynic" and "cup"-style vowels as "mop".
    "AA": ["AH"],
    "AE": ["AE"],
    "AH": ["UH"],
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


def _apply_klatt_duration(
    phonemes_with_info: list[dict],
) -> list[dict]:
    """Apply Klatt's multiplicative duration rules to phoneme sequence.

    Each entry in phonemes_with_info is a dict with keys:
        'base_votrax': str, 'stress': int, 'arpabet_base': str,
        'votrax_names': list[str], 'is_word_final': bool,
        'is_last_word': bool, 'word_syllable_count': int,
        'prev_arpabet': str|None, 'next_arpabet': str|None

    Modifies each entry to add 'duration_mult' (float).
    """
    n = len(phonemes_with_info)
    for i, info in enumerate(phonemes_with_info):
        mult = 1.0
        stress = info['stress']
        base = info['arpabet_base']
        is_vowel = base in _ARPABET_VOWELS

        # Rule 1: Pre-pausal (before pause or end)
        if info['is_word_final'] and info['is_last_word']:
            mult *= 1.4

        # Rule 4/5: Stress
        if stress == 1:
            mult *= 1.3
        elif stress == 0 and is_vowel:
            mult *= 0.75

        # Rule 6: Vowel before voiceless consonant
        if is_vowel and info['next_arpabet'] in _VOICELESS:
            mult *= 0.75

        # Rule 8: Polysyllabic shortening
        if is_vowel and info.get('word_syllable_count', 1) >= 3:
            mult *= 0.85

        # Rule 3: Non-phrase-final shortening
        if not (info['is_word_final'] and info['is_last_word']):
            mult *= 0.85

        # Rule 9: Non-initial consonant
        if not is_vowel and i > 0 and not info.get('is_word_initial', True):
            mult *= 0.85

        info['duration_mult'] = mult

    return phonemes_with_info


def _select_variant_by_duration(base_votrax: str, duration_mult: float) -> str:
    """Select vowel variant whose relative duration best matches the target.

    Duration multiplier maps to variant index:
    <0.8 → shortest, 0.8-1.0 → short-mid, 1.0-1.2 → mid-long, >1.2 → longest
    """
    variants = _VOWEL_VARIANTS.get(base_votrax)
    if variants is None:
        return base_votrax

    n = len(variants)
    if duration_mult < 0.8:
        idx = 0
    elif duration_mult < 1.0:
        idx = min(1, n - 1)
    elif duration_mult < 1.2:
        idx = min(n - 2, n - 1)
    else:
        idx = n - 1
    return variants[idx]


def _apply_declination(result: list[tuple[int, int]], pa0_code: int, stop_code: int):
    """Apply F0 declination: position-dependent inflection offset.

    First third trends higher (+1), last third trends lower (-1), clamped 0-3.
    """
    # Count non-pause phonemes
    non_pause = [(i, c, inf) for i, (c, inf) in enumerate(result)
                 if c != pa0_code and c != stop_code]
    if len(non_pause) < 3:
        return

    n = len(non_pause)
    first_third = n // 3
    last_third_start = n - (n // 3)

    for pos, (idx, code, inf) in enumerate(non_pause):
        if pos < first_third:
            new_inf = min(inf + 1, 3)
        elif pos >= last_third_start:
            new_inf = max(inf - 1, 0)
        else:
            new_inf = inf
        result[idx] = (code, new_inf)


def _apply_question_intonation(result: list[tuple[int, int]],
                                pa0_code: int, stop_code: int,
                                last_word_start: int):
    """Apply gradual rising intonation over the final word for questions.

    Instead of abruptly setting the last 3 phonemes to inflection 3,
    ramp inflection up progressively across the final word's phonemes.
    """
    # Find phonemes belonging to the last word (from last_word_start to end)
    word_phones = []
    for j in range(last_word_start, len(result)):
        code, inf = result[j]
        if code != pa0_code and code != stop_code:
            word_phones.append(j)

    if not word_phones:
        return

    n = len(word_phones)
    for k, idx in enumerate(word_phones):
        # Progressive ramp: 0 → 1 → 2 → 3 across the word
        target_inf = min(3, 1 + (k * 3) // max(n, 1))
        code, _ = result[idx]
        result[idx] = (code, target_inf)


def _fujisaki_contour(result: list[tuple[int, int]],
                       pa0_code: int, stop_code: int,
                       stressed_positions: list[int]):
    """Apply Fujisaki phrase+accent F0 model.

    ln(F0(t)) = ln(Fb) + Σ Ap*Gp(t) + Σ Aa*Ga(t)
    Maps continuous F0 to SC-01A's 4 inflection levels.

    Args:
        result: List of (code, inflection) tuples to modify in-place.
        pa0_code: Pause phoneme code.
        stop_code: Stop phoneme code.
        stressed_positions: Indices into result where primary stress occurs.
    """
    non_pause = [(i, c) for i, (c, _) in enumerate(result)
                 if c != pa0_code and c != stop_code]
    if len(non_pause) < 3:
        return

    n = len(non_pause)

    # Fujisaki parameters
    alpha = 2.0    # phrase time constant (rad/s)
    Ap = 0.4       # phrase command magnitude
    beta = 20.0    # accent time constant (rad/s)
    Aa = 0.3       # accent command magnitude

    # Time in arbitrary units (each phoneme ~ 1 unit)
    f0_contour = []
    for pos in range(n):
        t = float(pos)

        # Phrase component: Gp(t) = alpha^2 * t * exp(-alpha*t)
        # Phrase onset at t=0
        gp = (alpha ** 2) * t * math.exp(-alpha * t) if t > 0 else 0.0
        phrase = Ap * gp

        # Accent components: one per stressed position
        accent = 0.0
        for sp in stressed_positions:
            # Find the position index of this stressed phoneme
            sp_pos = None
            for k, (idx, _) in enumerate(non_pause):
                if idx == sp:
                    sp_pos = k
                    break
            if sp_pos is None:
                continue

            # Step response of 2nd-order: Ga(t) = 1 - (1 + beta*dt)*exp(-beta*dt)
            dt = t - sp_pos
            if dt >= 0:
                ga = 1.0 - (1.0 + beta * dt) * math.exp(-beta * dt)
            else:
                ga = 0.0
            # Accent active for ~2 phoneme units
            if dt > 2.0:
                # Accent offset
                dt_off = dt - 2.0
                ga -= 1.0 - (1.0 + beta * dt_off) * math.exp(-beta * dt_off)
            accent += Aa * ga

        f0_val = phrase + accent
        f0_contour.append(f0_val)

    if not f0_contour:
        return

    # Map continuous F0 contour to 0-3 inflection levels
    min_f0 = min(f0_contour)
    max_f0 = max(f0_contour)
    f0_range = max_f0 - min_f0

    for pos, (idx, code) in enumerate(non_pause):
        if f0_range > 0:
            normalized = (f0_contour[pos] - min_f0) / f0_range
        else:
            normalized = 0.5
        inf = min(3, int(normalized * 4))
        result[idx] = (code, inf)


class VotraxTTS:
    """Text-to-speech engine using CMU dict and the Votrax SC-01A emulator.

    Args:
        enhanced: Enable enhanced prosody (Klatt duration rules, vowel variant
                  selection, F0 declination, Fujisaki model). Default False.
        dc_block: Forwarded to :class:`VotraxSynthesizer`. Applies a 20 Hz DC
                  blocker to the output. The SC-01A AO pin is DC-biased per the
                  1980 datasheet, so enabling this is recommended for file
                  export. Default False to preserve historical output.
        radiation_filter: Forwarded to :class:`VotraxSynthesizer`. Applies a
                  first-difference +6 dB/oct radiation filter. Default False.
        master_clock: Forwarded to :class:`VotraxSynthesizer`. Default 720 000.
        fx_fudge: Forwarded to :class:`VotraxSynthesizer`. Default 150/4000
                  (authentic).
    """

    def __init__(
        self,
        enhanced: bool = False,
        dc_block: bool = True,
        radiation_filter: bool = False,
        master_clock: float = 720_000.0,
        fx_fudge: float = 150.0 / 4000.0,
        closure_strength: float = 1.0,
        enhanced_dsp: bool = False,
        rd: float = 1.0,
    ):
        self._enhanced = enhanced
        self._synth = VotraxSynthesizer(
            dc_block=dc_block,
            radiation_filter=radiation_filter,
            master_clock=master_clock,
            fx_fudge=fx_fudge,
            closure_strength=closure_strength,
            enhanced_dsp=enhanced_dsp,
            rd=rd,
        )

    def text_to_phonemes(self, text: str) -> list[tuple[int, int]]:
        """Convert text to a list of (votrax_code, inflection) tuples.

        Words are looked up in the CMU Pronouncing Dictionary.
        Unknown words are spelled out letter-by-letter.
        PA0 pauses are inserted between words, and STOP at the end.

        Sentence-level prosody:
        - Questions get rising inflection on final word (gradual in enhanced)
        - Statements get falling inflection on final phonemes
        - Pre-pausal lengthening: extra PA0 before final pause

        Enhanced mode additionally applies:
        - F0 declination across the sentence
        - Klatt duration rules (variant selection by computed duration)
        - Fujisaki phrase/accent F0 model
        """
        words = _tokenize(text)
        if not words:
            return []

        sentence_type = _detect_sentence_type(text)

        result = []
        pa0_code = name_to_code("PA0")
        stop_code = name_to_code("STOP")
        last_word_start = 0
        stressed_positions = []

        if self._enhanced:
            # Enhanced path: gather phoneme info for Klatt duration + Fujisaki
            for i, word in enumerate(words):
                is_last = (i == len(words) - 1)
                last_word_start = len(result)
                arpabet = _word_to_arpabet(word)
                if arpabet is not None:
                    parsed = [_strip_stress(phone) for phone in arpabet]

                    # Count syllables in this word
                    syllable_count = sum(1 for base, stress in parsed
                                         if base in _ARPABET_VOWELS)

                    for j, (base, stress) in enumerate(parsed):
                        inflection = _stress_to_inflection(stress)
                        votrax_names = ARPABET_TO_VOTRAX.get(base)
                        if votrax_names is None:
                            continue

                        prev_base = parsed[j - 1][0] if j > 0 else None
                        next_base = parsed[j + 1][0] if j < len(parsed) - 1 else None
                        is_word_final = (j == len(parsed) - 1)
                        next_for_variant = None if (is_word_final and is_last) else next_base

                        # Build info dict for Klatt
                        info = {
                            'base_votrax': votrax_names[0],
                            'stress': stress,
                            'arpabet_base': base,
                            'votrax_names': votrax_names,
                            'is_word_final': is_word_final,
                            'is_last_word': is_last,
                            'word_syllable_count': syllable_count,
                            'prev_arpabet': prev_base,
                            'next_arpabet': next_base,
                            'is_word_initial': (j == 0),
                        }
                        _apply_klatt_duration([info])

                        for vname in votrax_names:
                            if vname == votrax_names[0]:
                                # Use duration-based variant selection
                                vname = _select_variant_by_duration(
                                    vname, info['duration_mult'])
                            code = name_to_code(vname)
                            if stress == 1:
                                stressed_positions.append(len(result))
                            result.append((code, inflection))
                else:
                    result.extend(_spell_word(word))

                if i < len(words) - 1:
                    result.append((pa0_code, 0))

            # Apply Fujisaki F0 model
            if stressed_positions:
                _fujisaki_contour(result, pa0_code, stop_code,
                                  stressed_positions)

            # Apply declination on top
            _apply_declination(result, pa0_code, stop_code)

            # Sentence-final contour
            if result and sentence_type == "question":
                _apply_question_intonation(result, pa0_code, stop_code,
                                            last_word_start)
            elif result and sentence_type == "statement":
                for j in range(len(result) - 1, max(len(result) - 3, -1), -1):
                    code, inf = result[j]
                    if code != pa0_code and code != stop_code:
                        result[j] = (code, 0)

        else:
            # Standard path (unchanged)
            for i, word in enumerate(words):
                is_last = (i == len(words) - 1)
                last_word_start = len(result)
                arpabet = _word_to_arpabet(word)
                if arpabet is not None:
                    result.extend(arpabet_to_votrax(arpabet, is_last_word=is_last))
                else:
                    result.extend(_spell_word(word))

                if i < len(words) - 1:
                    result.append((pa0_code, 0))

            # Apply sentence-final contour
            if result and sentence_type == "question":
                for j in range(len(result) - 1, max(len(result) - 4, -1), -1):
                    code, inf = result[j]
                    if code != pa0_code and code != stop_code:
                        result[j] = (code, 3)
            elif result and sentence_type == "statement":
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
        self._synth.write_wav(audio, filename, target_rate=sample_rate)
