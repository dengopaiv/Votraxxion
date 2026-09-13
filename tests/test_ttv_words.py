"""Letter names and the number reader in the English front end.

Both are what a screen reader leans on hardest: NVDA spells a letter for every
character review keystroke, and reads numbers in every clock, page count and
version string. The expectations are phone names rather than hashes, so a
failure says what was heard instead of just that something moved.

Skipped unless the library has been built (nvda-addon/package.py).
"""

import ctypes
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

DLL = ROOT / "nvda-addon" / "addon" / "synthDrivers" / (
    "votraxNative-x64.dll" if ctypes.sizeof(ctypes.c_void_p) == 8
    else "votraxNative-x86.dll"
)

pytestmark = pytest.mark.skipif(
    not DLL.is_file(),
    reason=f"{DLL.name} not built - run nvda-addon/package.py",
)

PAUSES = {"PA0", "PA1"}


@pytest.fixture(scope="module")
def lib():
    import goldens
    return goldens.bind(str(DLL))


def _names(lib, fn, text, keep_pauses=False):
    buf = (ctypes.c_ubyte * 8192)()
    n = fn(text.encode("ascii"), buf, len(buf))
    names = [lib.vx_phone_name(buf[i] & 0x3F).decode() for i in range(n)]
    if not keep_pauses:
        names = [p for p in names if p not in PAUSES]
    return " ".join(names)


def spell(lib, text):
    return _names(lib, lib.ttv_spell, text)


def say(lib, text, keep_pauses=False):
    return _names(lib, lib.ttv_translate_flat, text, keep_pauses)


class TestLetterNames:
    """The seven names the 1985 table had wrong. Geczy's votraxsc01-nvda
    corrected O, U and S; H, Q, W and Y were found checking all 26 after."""

    @pytest.mark.parametrize("letter, expected, heard_before", [
        ("o", "O1 U1",             "AH: 'ah'"),
        ("u", "Y1 IU U",           "UH W: 'uh-w'"),
        ("s", "EH S",              "EH Z: a voiced, buzzy 'ezz'"),
        ("q", "K Y1 IU U",         "K W: 'kw'"),
        ("w", "D UH B L Y1 IU U",  "a trailing W: 'double-you-w'"),
        ("y", "W AH E1",           "a trailing E: 'why-ee'"),
        ("h", "A AY T CH",         "a doubled T before CH"),
    ])
    def test_corrected(self, lib, letter, expected, heard_before):
        assert spell(lib, letter) == expected, "was " + heard_before

    def test_case_does_not_matter(self, lib):
        for c in "abcdefghijklmnopqrstuvwxyz":
            assert spell(lib, c) == spell(lib, c.upper())

    def test_unchanged_letters_are_unchanged(self, lib):
        assert spell(lib, "a") == "A AY"
        assert spell(lib, "b") == "B E"
        assert spell(lib, "r") == "AH R"


class TestCardinals:
    @pytest.mark.parametrize("text, expected", [
        ("0", "Z I R O1 U1"),
        ("7", "S EH V UH2 N"),
        ("42", "F O R T E T IU U"),
        ("3000005", "TH R E M I L E UH2 N AH N D F AH E1 V"),
        ("100", "W UH N H UH N D R EH D"),
        ("1024", "W UH N TH AH O1 Z AE N D AH N D T W EH N T E F O1 U1 R"),
        ("1984", "N AH E1 N T E N H UH N D R EH D A AY T E F O1 U1 R"),
        ("2000", "T IU U TH AH O1 Z AE N D"),
        ("1000000", "W UH N M I L E UH2 N"),
    ])
    def test_read_as_quantities(self, lib, text, expected):
        assert say(lib, text) == expected

    def test_thousands_separators_join(self, lib):
        assert say(lib, "1,000,000") == say(lib, "1000000")
        assert say(lib, "12,345") == say(lib, "12345")

    def test_a_comma_list_stays_a_list(self, lib):
        """'1,2,3' is three numbers with pauses, not one hundred twenty-three."""
        assert say(lib, "1,2,3", keep_pauses=True).count("PA1") == 2
        assert say(lib, "12,34") != say(lib, "1234")

    def test_leading_zero_is_an_identifier(self, lib):
        assert say(lib, "007") == " ".join(
            [say(lib, "0"), say(lib, "0"), say(lib, "7")])

    def test_over_twelve_digits_is_an_identifier(self, lib):
        digits = "1234567890123"
        assert say(lib, digits) == " ".join(say(lib, d) for d in digits)
        # Twelve is still a quantity: nine hundred ninety-nine billion ...
        assert say(lib, "999999999999").count("B I L E UH2 N") == 1


class TestOrdinalsDecimalsMoney:
    @pytest.mark.parametrize("text, cardinal_prefix", [
        ("1st", None), ("2nd", None), ("3rd", None),
        ("11th", "E I3 L UH3 EH V EH N"),      # Wasser read this "eleven T H"
        ("21st", "T W EH N T E"),
        ("100th", "W UH N H UH N D R EH D"),
    ])
    def test_ordinals_take_their_suffix(self, lib, text, cardinal_prefix):
        got = say(lib, text)
        assert got.endswith("TH") or got.endswith("S T") or got.endswith("D"), got
        if cardinal_prefix:
            assert got.startswith(cardinal_prefix)

    def test_first_second_third(self, lib):
        assert say(lib, "1st") == "F ER S T"
        assert say(lib, "2nd") == "S EH K UH N D"

    def test_a_decimal_point_is_spoken_not_paused(self, lib):
        got = say(lib, "3.14", keep_pauses=True)
        assert got.startswith("TH R E PA0 P O1 E1 N T")
        assert "PA1" not in got

    def test_a_version_string_repeats_point(self, lib):
        assert say(lib, "1.2.3").count("P O1 E1 N T") == 2

    def test_a_sentence_ending_in_a_number_still_pauses(self, lib):
        assert say(lib, "Section 7.", keep_pauses=True).endswith("PA1 PA1")

    @pytest.mark.parametrize("text, expected", [
        ("$1", "W UH N D AH L UH3 ER"),
        ("$5", "F AH E1 V D AH L UH3 ER Z"),
        ("$4.20", "F O1 U1 R D AH L UH3 ER Z AH N D T W EH N T E S EH N T S"),
        ("$0.01", "W UH N S EH N T"),
        ("$1.5", "W UH N P O1 E1 N T F AH E1 V D AH L UH3 ER Z"),
    ])
    def test_money(self, lib, text, expected):
        assert say(lib, text) == expected

    def test_a_lone_dollar_sign_is_not_money(self, lib):
        assert say(lib, "$") == ""
