"""Sentence prosody, through the native library.

The SC-01 has four pitch levels and nothing between them. What the front end
does with that budget is assign a level per phone from its position in its own
sentence, with the final punctuation choosing the shape. These tests pin the
shapes down and, more importantly, pin down the invariants — that levels stay
in range, that packing is reversible, and that a contour never leaks across a
sentence boundary.

Skipped unless the DLL has been built (nvda-addon/package.py).
"""

import ctypes
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DLL = ROOT / "nvda-addon" / "addon" / "synthDrivers" / (
    "votraxNative-x64.dll" if ctypes.sizeof(ctypes.c_void_p) == 8
    else "votraxNative-x86.dll")

pytestmark = pytest.mark.skipif(
    not DLL.is_file(),
    reason=f"{DLL.name} not built - run nvda-addon/package.py",
)

NEUTRAL = 1          # VX_NEUTRAL_INFLECTION


@pytest.fixture(scope="module")
def lib():
    lib = ctypes.CDLL(str(DLL))
    p = ctypes.c_void_p
    lib.vx_create.restype = p
    lib.vx_create.argtypes = [ctypes.c_int, ctypes.c_uint]
    for name, res, args in (
        ("vx_destroy", None, [p]),
        ("vx_inflection", None, [p, ctypes.c_ubyte]),
        ("vx_get_inflection", ctypes.c_int, [p]),
        ("ttv_translate", ctypes.c_int, [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]),
        ("ttv_translate_flat", ctypes.c_int, [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]),
        ("ttv_spell", ctypes.c_int, [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]),
        ("vx_phone_name", ctypes.c_char_p, [ctypes.c_int]),
    ):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    return lib


def translate(lib, text, flat=False, spell=False):
    buf = ctypes.create_string_buffer(8192)
    fn = (lib.ttv_spell if spell else
          lib.ttv_translate_flat if flat else lib.ttv_translate)
    n = fn(text.encode("utf-8"), buf, 8192)
    return bytes(buf.raw[:n])


def phones(packed):
    return [b & 0x3F for b in packed]


def levels(packed):
    return [(b >> 6) & 3 for b in packed]


def voiced_levels(packed):
    """Levels of the non-pause phones — the ones that carry pitch."""
    return [(b >> 6) & 3 for b in packed if (b & 0x3F) not in (0x03, 0x3E)]


class TestPacking:
    def test_phone_survives_packing(self, lib):
        """The contour must not disturb which sounds are produced."""
        text = "The system is ready."
        assert phones(translate(lib, text)) == phones(translate(lib, text, flat=True))

    def test_flat_really_is_flat(self, lib):
        assert set(levels(translate(lib, "The system is ready.", flat=True))) == {0}

    def test_levels_stay_in_range(self, lib):
        for text in ("A statement.", "A question?", "An exclamation!",
                     "No terminator", "", "...", "12345"):
            assert all(0 <= v <= 3 for v in levels(translate(lib, text)))

    def test_unpacked_bytes_are_plain_phone_codes(self, lib):
        """A byte with clear top bits is just a phone, which is what makes
        packed and unpacked streams interchangeable."""
        for b in translate(lib, "hello", flat=True):
            assert b == b & 0x3F
            assert lib.vx_phone_name(b) is not None


class TestContours:
    def test_statement_falls(self, lib):
        v = voiced_levels(translate(lib, "The system is ready now."))
        assert v[0] > v[-1]
        assert v[-1] == 0

    def test_question_rises(self, lib):
        v = voiced_levels(translate(lib, "Is the system ready now?"))
        assert v[-1] > v[0]
        assert v[-1] == 3

    def test_exclamation_starts_high_and_falls(self, lib):
        v = voiced_levels(translate(lib, "The system is ready now!"))
        assert v[0] == 3
        assert v[-1] < v[0]

    def test_same_words_differ_by_punctuation(self, lib):
        """The contour is the only thing punctuation changes."""
        statement = translate(lib, "The system is ready.")
        question = translate(lib, "The system is ready?")
        assert phones(statement) == phones(question)
        assert levels(statement) != levels(question)

    def test_contour_is_monotonic(self, lib):
        """Each shape moves one way only — no wobble inside a sentence."""
        falling = voiced_levels(translate(lib, "The system is ready now."))
        rising = voiced_levels(translate(lib, "Is the system ready now?"))
        assert falling == sorted(falling, reverse=True)
        assert rising == sorted(rising)

    def test_no_terminator_reads_as_a_statement(self, lib):
        assert voiced_levels(translate(lib, "The system is ready")) == \
               voiced_levels(translate(lib, "The system is ready."))


class TestSentenceSplitting:
    def test_each_sentence_gets_its_own_contour(self, lib):
        """A paragraph must not be one long slide."""
        v = voiced_levels(translate(lib, "One is here. Two is here. Three is here."))
        # a single contour would end at its minimum and never come back up
        assert max(v[len(v) // 2:]) > min(v[:len(v) // 2])

    def test_contour_resets_after_each_terminator(self, lib):
        one = voiced_levels(translate(lib, "The system is ready."))
        two = voiced_levels(translate(lib, "The system is ready. The system is ready."))
        assert two[:len(one)] == one
        assert two[len(one):] == one

    def test_decimals_do_not_split_a_sentence(self, lib):
        """A full stop only ends a sentence when whitespace follows it."""
        v = voiced_levels(translate(lib, "It is 3.5 metres long."))
        assert v == sorted(v, reverse=True)     # one falling contour, not two

    def test_mixed_punctuation(self, lib):
        v = voiced_levels(translate(lib, "One. Two? Three!"))
        assert len(set(v)) > 1
        assert all(0 <= x <= 3 for x in v)


class TestPauses:
    def test_pauses_carry_the_surrounding_level(self, lib):
        """A level discontinuity across a pause would be a packing bug."""
        packed = translate(lib, "One two three four five.")
        previous = None
        for b in packed:
            phone, level = b & 0x3F, (b >> 6) & 3
            if phone in (0x03, 0x3E) and previous is not None:
                assert level == previous
            previous = level

    def test_pauses_do_not_consume_the_contour(self, lib):
        """Counting pauses would let a comma-heavy sentence spend its whole
        contour on silence."""
        plain = voiced_levels(translate(lib, "one two three four five six."))
        commas = voiced_levels(translate(lib, "one, two, three, four, five, six."))
        assert plain[-1] == commas[-1] == 0
        assert plain[0] == commas[0]


class TestSpelling:
    def test_spelling_is_flat(self, lib):
        """A spelled-out string is a list, not a sentence; a declination
        contour over it would imply a shape it does not have."""
        assert set(levels(translate(lib, "NVDA", spell=True))) == {NEUTRAL}


class TestBaseShiftsTheContour:
    def test_base_defaults_to_neutral(self, lib):
        chip = lib.vx_create(0, 0)
        try:
            assert lib.vx_get_inflection(chip) == NEUTRAL
        finally:
            lib.vx_destroy(chip)

    @pytest.mark.parametrize("base", [0, 1, 2, 3])
    def test_base_is_stored(self, lib, base):
        chip = lib.vx_create(0, 0)
        try:
            lib.vx_inflection(chip, base)
            assert lib.vx_get_inflection(chip) == base
        finally:
            lib.vx_destroy(chip)

    @pytest.mark.parametrize("base,packed,expected", [
        (1, 0, 0), (1, 1, 1), (1, 2, 2), (1, 3, 3),   # neutral: contour intact
        (2, 0, 1), (2, 3, 3),                          # transposed up, clipped
        (0, 3, 2), (0, 0, 0),                          # transposed down
        (3, 0, 2), (3, 3, 3),                          # clipped at the top
    ])
    def test_final_level_formula(self, base, packed, expected):
        """base + packed - NEUTRAL, clamped: what the scheduler computes."""
        assert max(0, min(3, base + packed - NEUTRAL)) == expected
